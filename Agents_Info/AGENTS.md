# Codex repository instructions — 3D LQG Analog

This repository supports exploratory mathematical research on three-dimensional analogues of Liouville quantum gravity (LQG), including Brownian mating-of-trees graphs, intrinsic random metrics, Gaussian fields, Gaussian multiplicative chaos (GMC), Liouville first-passage percolation (LFPP), and related diffusion/growth processes.

**The current task is whatever the user asks in the Codex conversation.** This file contains durable working rules, **not** an assignment or a fixed roadmap. Do not automatically start or continue a particular simulation just because it is described in a context document. If an older plan conflicts with the current prompt, follow the current prompt, while reporting any material mathematical or reproducibility concerns.

## Where to find context

- `RESEARCH_CONTEXT.md`: mathematical background, literature, definitions, conjectures, possible programs, and initial experiment ideas. Read the **relevant sections** when a task depends on them; it is reference material, not a standing task list. Treat historical priorities and experiment plans in it as *suggestions* unless the current prompt adopts them.
- `RESEARCH_LOG.md` (if present): prior experiments, findings, failures, source versions, and unresolved questions. Read when continuing earlier work. For substantial research or simulation tasks, create or update a concise dated entry with links to results; do not rewrite or erase previous observations.
- `papers/` (if present): locally saved source papers. Check the original paper—not only this project's summaries—when implementing a published construction or quoting a theorem. When permitted, retrieve a missing openly accessible source; otherwise give its exact URL and tell the user what is missing.
- Existing source code, tests, data, and Git history are the actual record of what has been implemented or run. Do not assume that a document's proposed work has been completed.

## Research rigor and provenance

1. Separate **proved theorems**, **published numerical evidence**, **physics heuristics**, **working conjectures**, and **new numerical findings**. Cite the primary source, paper version, equation/table/theorem, and assumptions whenever practical.
2. Respect distinctions among models and observables. In particular, three correlated Brownian coordinates do **not** prove 3-manifold topology; Budd–Castro graphs do **not** come with a canonical 3D log-correlated field or GMC measure; graph-distance exponents, LFPP distance exponents, and Hausdorff dimensions require explicit conversion before comparison.
3. Keep similarly named parameters distinct (Brownian covariance entries/angles, 2D or 3D LQG `gamma`, LFPP `xi` and distance exponent, graph Hausdorff dimension, and susceptibility exponents). Validate covariance matrices as positive definite or document singular limits.
4. Never assert that a matching exponent proves a common scaling limit, coupling, universality class, field law, or metric-measure identification. Test multiple independent observables and discuss identifiability and finite-size effects.
5. When using an approximation, biased sampling law, reweighting, surrogate field, or altered graph construction, clearly state what differs from the reference model. Verify the precise target ensemble and discretization from the paper before reporting replication of published numerical results.

## Coding, simulations, and verification

- Prefer small, documented, testable components; preserve working code and existing experimental outputs unless asked to replace them.
- Make experiments reproducible: store seeds, random-number generator, model definitions, parameter values/matrices, discretization, sample counts, software versions, run commands, and raw or summarized numerical data.
- Include mathematical/sanity tests (e.g., covariance positivity, graph invariants, baseline/special-case checks, and independent implementation checks when feasible).
- Quantify Monte Carlo uncertainty and fitting/window sensitivity; show finite-size trends, not just a single log–log fit. Record failed validations and numerical limitations.
- Save readable plots, tabular outputs, and a brief interpretation in task-appropriate folders. Distinguish completed runs from code that was merely written. Never fabricate data or results.
- Use computational resources responsibly: start with small validation runs before expensive sweeps. Report resource limitations honestly.
- Do not silently change unrelated files, remove previous results, rewrite this `AGENTS.md`, or edit `RESEARCH_CONTEXT.md` as part of an ordinary research task. Propose significant changes to foundational documents separately.

## External sources and network

Internet access depends on the Codex environment and permissions. Use existing local papers when available. If access is permitted, fetch authoritative open versions (prefer arXiv or author pages), save source metadata, and check for later versions or corrections. Do not assume this ChatGPT Project's uploads/chats are visible in the Codex workspace. If downloads are blocked, report the exact sources the user should place in `papers/`. Avoid bulk-committing PDFs to Git without the user's approval and respect source licenses.

## End-of-task handoff

For substantial work, summarize: (a) what was changed or implemented; (b) which tests and simulations actually ran; (c) where code, raw data, plots, and results were saved; (d) established conclusions versus tentative interpretations; and (e) next questions or failures. Update `RESEARCH_LOG.md` when appropriate. The next task will be specified through Codex chat, not through edits to this file.
