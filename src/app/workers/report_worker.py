"""Workers for generating draft reports and running refinements."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import frontmatter
from PySide6.QtCore import Signal

from src.app.core.project_manager import ProjectMetadata
from src.app.core.citations import strip_citation_tokens
from src.app.core.prompt_placeholders import format_prompt
from src.app.core.refinement_prompt import (
    read_generation_prompt,
    read_refinement_prompt,
    validate_generation_prompt,
    validate_refinement_prompt,
)
from src.app.core.report_template_sections import TemplateSection, load_template_sections
from src.app.core.report_prompt_context import (
    build_report_generation_placeholders,
    build_report_refinement_placeholders,
)
from src.config.observability import trace_operation
from src.common.markdown import (
    PromptReference,
    apply_frontmatter,
    build_document_metadata,
    compute_file_checksum,
)
from .llm_backend import (
    LLMExecutionBackend,
    LLMInvocationRequest,
    backend_route_name,
    backend_transport_name,
)
from .stage_contracts import (
    ReportDraftStageInput,
    ReportRefineStageInput,
    stage_trace_attributes,
)

from .report_common import ReportWorkerBase


class DraftReportWorker(ReportWorkerBase):
    """Generate a draft report from selected project inputs."""

    progress = Signal(int, str)
    progress_detail = Signal(object)
    log_message = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    cost_calculated = Signal(float, str, str)

    def __init__(
        self,
        *,
        project_dir: Path,
        inputs: Sequence[tuple[str, str]],
        provider_id: str,
        model: str,
        custom_model: Optional[str],
        context_window: Optional[int],
        template_path: Path,
        transcript_path: Optional[Path],
        generation_user_prompt_path: Path,
        generation_system_prompt_path: Path,
        metadata: ProjectMetadata,
        use_reasoning: bool = False,
        reasoning: Mapping[str, Any] | None = None,
        max_report_tokens: int = 60_000,
        placeholder_values: Mapping[str, str] | None = None,
        project_name: str = "",
        estimate_summary: Mapping[str, object] | None = None,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        super().__init__(
            worker_name="report-draft",
            project_dir=project_dir,
            inputs=inputs,
            provider_id=provider_id,
            model=model,
            custom_model=custom_model,
            context_window=context_window,
            use_reasoning=use_reasoning,
            reasoning=reasoning,
            metadata=metadata,
            placeholder_values=placeholder_values,
            project_name=project_name,
            max_report_tokens=max_report_tokens,
            llm_backend=llm_backend,
        )
        self._estimate_summary = dict(estimate_summary or {})
        self._template_path = Path(template_path)
        self._transcript_path = Path(transcript_path) if transcript_path else None
        self._generation_user_prompt_path = Path(generation_user_prompt_path).expanduser()
        self._generation_system_prompt_path = Path(generation_system_prompt_path).expanduser()

    # ------------------------------------------------------------------
    # QRunnable implementation
    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - exercised in tests
        try:
            if not self._template_path.exists():
                raise FileNotFoundError(f"Report template not found: {self._template_path}")
            if not self._generation_user_prompt_path.exists():
                raise FileNotFoundError(
                    f"Generation user prompt not found: {self._generation_user_prompt_path}"
                )
            if not self._generation_system_prompt_path.exists():
                raise FileNotFoundError(
                    f"Generation system prompt not found: {self._generation_system_prompt_path}"
                )
            if not self._inputs and not self._transcript_path:
                raise RuntimeError(
                    "Select at least one input or provide a transcript before generating a draft"
                )
            if self._transcript_path and not self._transcript_path.exists():
                raise FileNotFoundError(f"Transcript not found: {self._transcript_path}")

            timestamp = datetime.now(timezone.utc)
            report_dir = self._project_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            base_name = timestamp.strftime("report-%Y%m%d-%H%M%S")
            draft_path = report_dir / f"{base_name}-draft.md"
            manifest_path = report_dir / f"{base_name}-draft.manifest.json"
            inputs_path = report_dir / f"{base_name}-inputs.md"

            placeholder_map = self._placeholder_map()

            transcript_note = " and transcript" if self._transcript_path else ""
            self.log_message.emit(
                f"Preparing draft run with {len(self._inputs)} input(s){transcript_note}."
            )
            self.progress.emit(5, "Reading inputs…")
            self._emit_progress_detail(
                run_kind="report_draft",
                phase="preparing",
                label="Reading inputs",
                percent=5,
                detail=f"{len(self._inputs)} selected input(s)",
            )
            combined_content, inputs_metadata = self._combine_inputs()
            inputs_metadata = list(inputs_metadata)
            citation_appendix, citation_label_mapping = self._build_report_citation_appendix(inputs_metadata)
            input_sources = self._input_sources(inputs_metadata)
            inputs_payload = build_document_metadata(
                project_path=self._project_dir,
                generator="draft_report_worker",
                created_at=timestamp,
                sources=input_sources,
                extra={
                    "document_type": "report-inputs",
                    "input_count": len(input_sources),
                    "categories": sorted({ref.role for ref in input_sources if ref.role}),
                },
            )
            inputs_content = apply_frontmatter(combined_content, inputs_payload, merge_existing=True)
            inputs_path.write_text(inputs_content, encoding="utf-8")
            inputs_checksum = compute_file_checksum(inputs_path)
            self.log_message.emit(f"Combined inputs written to {inputs_path.name}.")

            generation_user_prompt = read_generation_prompt(self._generation_user_prompt_path)
            validate_generation_prompt(generation_user_prompt)

            generation_system_prompt = self._generation_system_prompt_path.read_text(
                encoding="utf-8"
            ).strip()
            if not generation_system_prompt:
                raise RuntimeError(
                    "Generation system prompt cannot be empty. Update the selected file."
                )

            generation_system_prompt = format_prompt(
                generation_system_prompt,
                placeholder_map,
            )
            generation_system_prompt = self._append_citation_appendix(
                generation_system_prompt,
                citation_appendix,
            )

            sections = load_template_sections(self._template_path)
            if not sections:
                raise RuntimeError("Template does not contain any sections to process")

            transcript_text = ""
            if self._transcript_path:
                transcript_text = self._transcript_path.read_text(encoding="utf-8").strip()
            additional_documents = combined_content.strip()

            self.log_message.emit(
                f"Generating draft content across {len(sections)} template section(s)…"
            )
            self._emit_progress_detail(
                run_kind="report_draft",
                phase="generating_sections",
                label="Generating draft sections",
                percent=10,
                detail=f"{len(sections)} section(s)",
            )

            section_outputs = self._generate_section_outputs(
                sections=sections,
                user_prompt_template=generation_user_prompt,
                additional_documents=additional_documents,
                transcript_text=transcript_text,
                system_prompt=generation_system_prompt,
                placeholder_map=placeholder_map,
            )

            draft_content = self._combine_section_outputs(section_outputs)
            if not draft_content.strip():
                raise RuntimeError("Section generation produced empty draft content")

            draft_body = self._format_draft_header(draft_content)
            draft_sources = list(input_sources)
            combined_inputs_ref = self._file_source(
                inputs_path, role="combined-inputs", checksum=inputs_checksum
            )
            if combined_inputs_ref:
                draft_sources.append(combined_inputs_ref)
            template_ref = self._optional_source(self._template_path, role="template")
            if template_ref:
                draft_sources.append(template_ref)
            transcript_ref = self._optional_source(self._transcript_path, role="transcript")
            if transcript_ref:
                draft_sources.append(transcript_ref)
            generation_prompts = [
                self._prompt_reference(self._generation_user_prompt_path, role="generation-user"),
                self._prompt_reference(self._generation_system_prompt_path, role="generation-system"),
            ]
            draft_payload = build_document_metadata(
                project_path=self._project_dir,
                generator="draft_report_worker",
                created_at=datetime.now(timezone.utc),
                sources=draft_sources,
                prompts=[ref for ref in generation_prompts if ref.to_dict()],
                extra={
                    "document_type": "report-draft",
                    "section_count": len(section_outputs),
                    "placeholders": dict(placeholder_map),
                },
            )
            draft_content_prepared = apply_frontmatter(draft_body, draft_payload, merge_existing=True)
            draft_path.write_text(draft_content_prepared, encoding="utf-8")
            draft_citation_stats = self._record_output_citations(
                output_path=draft_path,
                output_text=draft_body,
                generator="draft_report_worker",
                label_mapping=citation_label_mapping,
            )

            manifest = self._build_draft_manifest(
                timestamp=timestamp,
                draft_path=draft_path,
                inputs_path=inputs_path,
                inputs_metadata=inputs_metadata,
                sections=section_outputs,
                template_path=self._template_path,
                transcript_path=self._transcript_path,
                generation_user_prompt=self._generation_user_prompt_path,
                generation_system_prompt=self._generation_system_prompt_path,
                citation_stats=draft_citation_stats,
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.log_message.emit(f"Draft manifest written to {manifest_path.name}.")

            result = {
                "timestamp": timestamp.isoformat(),
                "draft_path": str(draft_path),
                "manifest_path": str(manifest_path),
                "inputs_path": str(inputs_path),
                "provider": self._provider_id,
                "model": self._custom_model or self._model,
                "custom_model": self._custom_model,
                "context_window": self._context_window,
                "use_reasoning": self._use_reasoning,
                "reasoning": self._reasoning.to_dict(),
                "inputs": [item[1] for item in self._inputs],
                "template_path": str(self._template_path),
                "transcript_path": str(self._transcript_path) if self._transcript_path else None,
                "generation_user_prompt": str(self._generation_user_prompt_path),
                "generation_system_prompt": str(self._generation_system_prompt_path),
                "section_count": len(section_outputs),
                "usage": self._usage_summary(),
                "cost": self._total_cost(),
                "cost_estimate": dict(self._estimate_summary) if self._estimate_summary else None,
            }
            if result["cost"] is not None:
                self.cost_calculated.emit(float(result["cost"]), self._provider_id, "report_draft")
            self.progress.emit(100, "Draft generated")
            self._emit_progress_detail(
                run_kind="report_draft",
                phase="completed",
                label="Draft generated",
                percent=100,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_section_outputs(
        self,
        *,
        sections: Sequence[TemplateSection],
        user_prompt_template: str,
        additional_documents: str,
        transcript_text: str,
        system_prompt: str,
        placeholder_map: Mapping[str, str],
    ) -> List[dict]:
        self.log_message.emit(self._llm_execution_summary())
        provider = self._create_provider()
        outputs: List[dict] = []

        total = len(sections)
        for index, section in enumerate(sections, start=1):
            generation_placeholders = build_report_generation_placeholders(
                base_placeholders=placeholder_map,
                template_section=section.body.strip(),
                section_title=section.title or "",
                transcript=transcript_text,
                additional_documents=additional_documents,
            )
            prompt = format_prompt(user_prompt_template, generation_placeholders)

            pct = 5 + int(60 * index / max(total, 1))
            self.progress.emit(pct, f"Generating section {index} of {total}: {section.title}")
            self._emit_progress_detail(
                run_kind="report_draft",
                phase="section_started",
                label=f"Generating section {index} of {total}",
                percent=pct,
                section_index=index,
                section_total=total,
                section_title=section.title,
            )
            stage_input = ReportDraftStageInput(
                section_index=index,
                section_total=total,
                section_title=section.title,
                transport=backend_transport_name(self._llm_backend),
                provider_id=self._provider_id,
                model=self._custom_model or self._model,
                reasoning=self._use_reasoning,
                gateway_route=backend_route_name(self._llm_backend),
                max_tokens=self._max_report_tokens,
                temperature=0.2,
            )
            trace_attributes = stage_trace_attributes(stage_input)
            section_trace_attributes = dict(trace_attributes)
            section_trace_attributes["llestrade.phase_name"] = "section"
            with trace_operation("report_draft.section", section_trace_attributes):
                with trace_operation("report_draft.invoke_llm", trace_attributes):
                    budget_evaluation = self._evaluate_request_budget(
                        provider,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_output_tokens=self._max_report_tokens,
                        temperature=0.2,
                    )
                    if not budget_evaluation.fits:
                        raise RuntimeError(
                            f"Prompt exceeds model input budget for report draft section '{section.title}': "
                            f"{budget_evaluation.input_tokens} tokens > {budget_evaluation.preflight_input_budget} budget"
                        )
                    response = self._llm_backend.invoke_response(
                        provider,
                        LLMInvocationRequest(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            model=self._custom_model or self._model,
                            model_settings=self._llm_backend.build_model_settings(
                                self._provider_id,
                                self._custom_model or self._model,
                                temperature=0.2,
                                max_tokens=self._max_report_tokens,
                                use_reasoning=self._use_reasoning,
                                reasoning=self._reasoning,
                            ),
                            input_tokens_limit=budget_evaluation.runtime_input_budget,
                        ),
                    )
                self._emit_progress_detail(
                    run_kind="report_draft",
                    phase="section_completed",
                    label=f"Completed section {index} of {total}",
                    percent=pct,
                    section_index=index,
                    section_total=total,
                    section_title=section.title,
                )
            self._record_response_usage(response)
            content = str(response.text or "").strip()
            if not content:
                raise RuntimeError(f"Generated section is empty: {section.title}")
            outputs.append(
                {
                    "title": section.title,
                    "prompt": prompt,
                    "content": content,
                }
            )
            self.log_message.emit(f"Section generated: {section.title}")

        return outputs

    def _combine_section_outputs(self, outputs: Sequence[dict]) -> str:
        combined_sections = []
        for payload in outputs:
            content = payload.get("content", "").strip()
            if content:
                combined_sections.append(content)
        return "\n\n".join(combined_sections)

    def _format_draft_header(self, content: str) -> str:
        metadata = self._metadata or ProjectMetadata(case_name="")
        subject = metadata.subject_name or metadata.case_name or "Unknown"
        dob = metadata.date_of_birth or "Unknown"
        generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
        header = (
            "# Integrated Report\n\n"
            f"**Subject**: {subject}\n\n"
            f"**Date of Birth**: {dob}\n\n"
            f"**Generated**: {generated}\n\n"
        )
        return header + content.strip() + "\n"

    def _build_draft_manifest(
        self,
        *,
        timestamp: datetime,
        draft_path: Path,
        inputs_path: Path,
        inputs_metadata: Sequence[dict],
        sections: Sequence[dict],
        template_path: Path,
        transcript_path: Optional[Path],
        generation_user_prompt: Path,
        generation_system_prompt: Path,
        citation_stats: object = None,
    ) -> Dict[str, object]:
        manifest: Dict[str, object] = {
            "version": 2,
            "run_type": "draft",
            "timestamp": timestamp.isoformat(),
            "provider": self._provider_id,
            "model": self._model,
            "custom_model": self._custom_model,
            "context_window": self._context_window,
            "use_reasoning": self._use_reasoning,
            "draft_path": str(draft_path),
            "inputs_path": str(inputs_path),
            "template_path": str(template_path),
            "transcript_path": str(transcript_path) if transcript_path else None,
            "generation_user_prompt": str(generation_user_prompt),
            "generation_system_prompt": str(generation_system_prompt),
            "inputs": list(inputs_metadata),
            "sections": [
                {
                    "title": payload.get("title"),
                    "content": payload.get("content"),
                }
                for payload in sections
            ],
            "usage": self._usage_summary(),
            "cost": self._total_cost(),
        }
        if self._estimate_summary:
            manifest["cost_estimate"] = dict(self._estimate_summary)
        if citation_stats is not None:
            manifest["citations"] = {
                "total": citation_stats.total,
                "valid": citation_stats.valid,
                "warning": citation_stats.warning,
                "invalid": citation_stats.invalid,
            }
        return manifest


class ReportRefinementWorker(ReportWorkerBase):
    """Refine an existing draft report into a variant."""

    progress = Signal(int, str)
    progress_detail = Signal(object)
    log_message = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    cost_calculated = Signal(float, str, str)

    def __init__(
        self,
        *,
        project_dir: Path,
        draft_path: Path,
        inputs: Sequence[tuple[str, str]],
        provider_id: str,
        model: str,
        custom_model: Optional[str],
        context_window: Optional[int],
        template_path: Optional[Path],
        transcript_path: Optional[Path],
        refinement_user_prompt_path: Path,
        refinement_system_prompt_path: Path,
        metadata: ProjectMetadata,
        use_reasoning: bool = False,
        reasoning: Mapping[str, Any] | None = None,
        max_report_tokens: int = 60_000,
        placeholder_values: Mapping[str, str] | None = None,
        project_name: str = "",
        estimate_summary: Mapping[str, object] | None = None,
        llm_backend: LLMExecutionBackend | None = None,
    ) -> None:
        super().__init__(
            worker_name="report-refine",
            project_dir=project_dir,
            inputs=inputs,
            provider_id=provider_id,
            model=model,
            custom_model=custom_model,
            context_window=context_window,
            use_reasoning=use_reasoning,
            reasoning=reasoning,
            metadata=metadata,
            placeholder_values=placeholder_values,
            project_name=project_name,
            max_report_tokens=max_report_tokens,
            llm_backend=llm_backend,
        )
        self._estimate_summary = dict(estimate_summary or {})
        self._draft_path = Path(draft_path)
        self._template_path = Path(template_path) if template_path else None
        self._transcript_path = Path(transcript_path) if transcript_path else None
        self._refinement_user_prompt_path = Path(refinement_user_prompt_path).expanduser()
        self._refinement_system_prompt_path = Path(refinement_system_prompt_path).expanduser()
        self._refine_usage: Optional[int] = None

    # ------------------------------------------------------------------
    # QRunnable implementation
    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - exercised in tests
        try:
            if not self._draft_path.exists():
                raise FileNotFoundError(f"Draft not found: {self._draft_path}")
            if not self._refinement_user_prompt_path.exists():
                raise FileNotFoundError(
                    f"Refinement user prompt not found: {self._refinement_user_prompt_path}"
                )
            if not self._refinement_system_prompt_path.exists():
                raise FileNotFoundError(
                    f"Refinement system prompt not found: {self._refinement_system_prompt_path}"
                )
            if self._template_path and not self._template_path.exists():
                raise FileNotFoundError(f"Template not found: {self._template_path}")
            if self._transcript_path and not self._transcript_path.exists():
                raise FileNotFoundError(f"Transcript not found: {self._transcript_path}")

            report_dir = self._draft_path.parent
            timestamp = datetime.now(timezone.utc)
            base_stem = self._draft_path.stem
            if base_stem.endswith("-draft"):
                base_stem = base_stem[:-6]
            refinement_token = timestamp.strftime("%Y%m%d-%H%M%S")
            refinement_base = f"{base_stem}-refine-{refinement_token}"
            refined_path = report_dir / f"{refinement_base}.md"
            reasoning_path = report_dir / f"{refinement_base}-reasoning.md"
            manifest_path = report_dir / f"{refinement_base}.manifest.json"
            inputs_path = report_dir / f"{refinement_base}-inputs.md"

            placeholder_map = self._placeholder_map()

            combined_content, inputs_metadata = self._combine_inputs()
            inputs_metadata = list(inputs_metadata)
            citation_appendix, citation_label_mapping = self._build_report_citation_appendix(inputs_metadata)
            input_sources = self._input_sources(inputs_metadata)
            if combined_content:
                inputs_payload = build_document_metadata(
                    project_path=self._project_dir,
                    generator="report_refinement_worker",
                    created_at=timestamp,
                    sources=input_sources,
                    extra={
                        "document_type": "report-refine-inputs",
                        "input_count": len(input_sources),
                        "categories": sorted({ref.role for ref in input_sources if ref.role}),
                    },
                )
                inputs_content = apply_frontmatter(
                    combined_content, inputs_payload, merge_existing=True
                )
                inputs_path.write_text(inputs_content, encoding="utf-8")
                inputs_checksum = compute_file_checksum(inputs_path)
                self.log_message.emit(f"Supplemental inputs written to {inputs_path.name}.")
            else:
                inputs_path = None
                inputs_checksum = None

            draft_document = frontmatter.loads(
                self._draft_path.read_text(encoding="utf-8")
            )
            draft_content = strip_citation_tokens(draft_document.content or "").strip()
            if not draft_content:
                raise RuntimeError("Draft content is empty; cannot run refinement.")

            refinement_user_prompt = read_refinement_prompt(self._refinement_user_prompt_path)
            validate_refinement_prompt(refinement_user_prompt)

            refinement_system_prompt = self._refinement_system_prompt_path.read_text(
                encoding="utf-8"
            ).strip()
            if not refinement_system_prompt:
                raise RuntimeError(
                    "Refinement system prompt cannot be empty. Update the selected file."
                )
            refinement_system_prompt = format_prompt(
                refinement_system_prompt,
                placeholder_map,
            )
            refinement_system_prompt = self._append_citation_appendix(
                refinement_system_prompt,
                citation_appendix,
            )

            template_raw = ""
            if self._template_path:
                template_raw = self._template_path.read_text(encoding="utf-8")
            transcript_raw = ""
            if self._transcript_path:
                transcript_raw = self._transcript_path.read_text(encoding="utf-8")

            self.log_message.emit("Refining draft report…")
            self.progress.emit(30, "Refining draft…")
            self._emit_progress_detail(
                run_kind="report_refine",
                phase="refining",
                label="Refining draft",
                percent=30,
            )
            refinement_placeholders = build_report_refinement_placeholders(
                base_placeholders=placeholder_map,
                draft_report=draft_content,
                template=template_raw,
                transcript=transcript_raw,
            )
            refine_prompt = format_prompt(refinement_user_prompt, refinement_placeholders)
            refined_content, reasoning_content = self._run_refinement(
                prompt=refine_prompt,
                system_prompt=refinement_system_prompt,
            )
            self.progress.emit(85, "Writing refinement outputs…")
            self._emit_progress_detail(
                run_kind="report_refine",
                phase="writing_outputs",
                label="Writing refinement outputs",
                percent=85,
            )

            refinement_prompts = [
                self._prompt_reference(self._refinement_user_prompt_path, role="refinement-user"),
                self._prompt_reference(self._refinement_system_prompt_path, role="refinement-system"),
            ]

            refined_sources = input_sources
            draft_ref = self._file_source(self._draft_path, role="draft")
            if draft_ref:
                refined_sources = list(refined_sources) + [draft_ref]
            template_ref = self._optional_source(self._template_path, role="template")
            if template_ref:
                refined_sources.append(template_ref)
            transcript_ref = self._optional_source(self._transcript_path, role="transcript")
            if transcript_ref:
                refined_sources.append(transcript_ref)
            inputs_ref = self._file_source(inputs_path, role="supplemental-inputs", checksum=inputs_checksum)
            if inputs_ref:
                refined_sources.append(inputs_ref)

            refined_payload = build_document_metadata(
                project_path=self._project_dir,
                generator="report_refinement_worker",
                created_at=datetime.now(timezone.utc),
                sources=refined_sources,
                prompts=[ref for ref in refinement_prompts if ref.to_dict()],
                extra={
                    "document_type": "report-refined",
                    "refinement_tokens": self._refine_usage,
                },
            )
            refined_content_prepared = apply_frontmatter(
                refined_content, refined_payload, merge_existing=True
            )
            refined_path.write_text(refined_content_prepared, encoding="utf-8")
            refined_checksum = compute_file_checksum(refined_path)
            refined_citation_stats = self._record_output_citations(
                output_path=refined_path,
                output_text=refined_content,
                generator="report_refinement_worker",
                label_mapping=citation_label_mapping,
            )
            self.log_message.emit(f"Refined report saved to {refined_path.name}.")

            reasoning_written: Optional[Path] = None
            if reasoning_content:
                reasoning_sources = list(refined_sources)
                refined_ref = self._file_source(
                    refined_path, role="refined", checksum=refined_checksum
                )
                if refined_ref:
                    reasoning_sources.append(refined_ref)
                reasoning_payload = build_document_metadata(
                    project_path=self._project_dir,
                    generator="report_refinement_worker",
                    created_at=datetime.now(timezone.utc),
                    sources=reasoning_sources,
                    prompts=[ref for ref in refinement_prompts if ref.to_dict()],
                    extra={"document_type": "report-reasoning"},
                )
                reasoning_prepared = apply_frontmatter(
                    reasoning_content, reasoning_payload, merge_existing=True
                )
                reasoning_path.write_text(reasoning_prepared, encoding="utf-8")
                reasoning_written = reasoning_path
                self.log_message.emit(f"Reasoning output saved to {reasoning_path.name}.")
            else:
                reasoning_path.unlink(missing_ok=True)  # ensure stale file removed if empty reasoning
                reasoning_written = None

            manifest = self._build_refinement_manifest(
                timestamp=timestamp,
                draft_path=self._draft_path,
                refined_path=refined_path,
                reasoning_path=reasoning_written,
                inputs_path=inputs_path,
                inputs_metadata=inputs_metadata,
                template_path=self._template_path,
                transcript_path=self._transcript_path,
                refinement_user_prompt=self._refinement_user_prompt_path,
                refinement_system_prompt=self._refinement_system_prompt_path,
                citation_stats=refined_citation_stats,
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.log_message.emit(f"Refinement manifest written to {manifest_path.name}.")

            result = {
                "timestamp": timestamp.isoformat(),
                "draft_path": str(self._draft_path),
                "refined_path": str(refined_path),
                "reasoning_path": str(reasoning_written) if reasoning_written else None,
                "manifest_path": str(manifest_path),
                "inputs_path": str(inputs_path) if inputs_path else None,
                "provider": self._provider_id,
                "model": self._custom_model or self._model,
                "custom_model": self._custom_model,
                "context_window": self._context_window,
                "use_reasoning": self._use_reasoning,
                "reasoning": self._reasoning.to_dict(),
                "inputs": [item[1] for item in self._inputs],
                "template_path": str(self._template_path) if self._template_path else None,
                "transcript_path": str(self._transcript_path) if self._transcript_path else None,
                "refinement_user_prompt": str(self._refinement_user_prompt_path),
                "refinement_system_prompt": str(self._refinement_system_prompt_path),
                "refinement_tokens": self._refine_usage,
                "usage": self._usage_summary(),
                "cost": self._total_cost(),
                "cost_estimate": dict(self._estimate_summary) if self._estimate_summary else None,
            }
            if result["cost"] is not None:
                self.cost_calculated.emit(float(result["cost"]), self._provider_id, "report_refinement")
            self.progress.emit(100, "Refinement completed")
            self._emit_progress_detail(
                run_kind="report_refine",
                phase="completed",
                label="Refinement completed",
                percent=100,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - defensive
            self.failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_refinement(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> tuple[str, Optional[str]]:
        self.log_message.emit(self._llm_execution_summary())
        provider = self._create_provider()
        stage_input = ReportRefineStageInput(
            transport=backend_transport_name(self._llm_backend),
            provider_id=self._provider_id,
            model=self._custom_model or self._model,
            reasoning=self._use_reasoning,
            gateway_route=backend_route_name(self._llm_backend),
            max_tokens=self._max_report_tokens,
            temperature=0.2,
        )
        trace_attributes = stage_trace_attributes(stage_input)
        phase_trace_attributes = dict(trace_attributes)
        phase_trace_attributes["llestrade.phase_name"] = "refining"
        with trace_operation("report_refine.phase", phase_trace_attributes):
            with trace_operation("report_refine.invoke_llm", trace_attributes):
                budget_evaluation = self._evaluate_request_budget(
                    provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_output_tokens=self._max_report_tokens,
                    temperature=0.2,
                )
                if not budget_evaluation.fits:
                    raise RuntimeError(
                        "Prompt exceeds model input budget for report refinement: "
                        f"{budget_evaluation.input_tokens} tokens > {budget_evaluation.preflight_input_budget} budget"
                    )
                response = self._llm_backend.invoke_response(
                    provider,
                    LLMInvocationRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=self._custom_model or self._model,
                        model_settings=self._llm_backend.build_model_settings(
                            self._provider_id,
                            self._custom_model or self._model,
                            temperature=0.2,
                            max_tokens=self._max_report_tokens,
                            use_reasoning=self._use_reasoning,
                            reasoning=self._reasoning,
                        ),
                        input_tokens_limit=budget_evaluation.runtime_input_budget,
                    ),
                )
        self._record_response_usage(response)
        content = str(response.text or "").strip()
        if not content:
            raise RuntimeError("Refinement step returned empty content")
        reasoning = response.thinking
        self._refine_usage = int(getattr(response.usage, "output_tokens", 0) or 0)
        return content + "\n", reasoning

    def _build_refinement_manifest(
        self,
        *,
        timestamp: datetime,
        draft_path: Path,
        refined_path: Path,
        reasoning_path: Optional[Path],
        inputs_path: Optional[Path],
        inputs_metadata: Sequence[dict],
        template_path: Optional[Path],
        transcript_path: Optional[Path],
        refinement_user_prompt: Path,
        refinement_system_prompt: Path,
        citation_stats: object = None,
    ) -> Dict[str, object]:
        manifest: Dict[str, object] = {
            "version": 2,
            "run_type": "refinement",
            "timestamp": timestamp.isoformat(),
            "provider": self._provider_id,
            "model": self._model,
            "custom_model": self._custom_model,
            "context_window": self._context_window,
            "use_reasoning": self._use_reasoning,
            "draft_path": str(draft_path),
            "refined_path": str(refined_path),
            "reasoning_path": str(reasoning_path) if reasoning_path else None,
            "inputs_path": str(inputs_path) if inputs_path else None,
            "template_path": str(template_path) if template_path else None,
            "transcript_path": str(transcript_path) if transcript_path else None,
            "refinement_user_prompt": str(refinement_user_prompt),
            "refinement_system_prompt": str(refinement_system_prompt),
            "inputs": list(inputs_metadata),
            "usage": {
                "refined_tokens": self._refine_usage,
                **self._usage_summary(),
            },
            "cost": self._total_cost(),
        }
        if self._estimate_summary:
            manifest["cost_estimate"] = dict(self._estimate_summary)
        if citation_stats is not None:
            manifest["citations"] = {
                "total": citation_stats.total,
                "valid": citation_stats.valid,
                "warning": citation_stats.warning,
                "invalid": citation_stats.invalid,
            }
        return manifest


__all__ = ["DraftReportWorker", "ReportRefinementWorker"]
