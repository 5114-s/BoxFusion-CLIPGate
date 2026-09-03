LaTeX Course Paper Package
==========================

Main source:
  course_paper_cross_modal_residual_mining.tex

Bibliography:
  course_paper_references.bib

Figures:
  figures/course_paper/figure1_method_pipeline.pdf
  figures/course_paper/figure2_protocol_results.pdf
  figures/course_paper/figure3_candidate_funnel.pdf

High-resolution PNG versions of the same figures are also included for Word
and presentation use.

Before submission, replace these placeholders in the .tex file:
  [Your Name]
  [Your Student ID]
  [Your School and University]

Compilation on Overleaf
-----------------------
1. Upload the ZIP package as a new project.
2. Set course_paper_cross_modal_residual_mining.tex as the main document.
3. Use pdfLaTeX. Overleaf runs BibTeX automatically.

Local compilation
-----------------
Run:

  pdflatex course_paper_cross_modal_residual_mining.tex
  bibtex course_paper_cross_modal_residual_mining
  pdflatex course_paper_cross_modal_residual_mining.tex
  pdflatex course_paper_cross_modal_residual_mining.tex

The current execution environment does not include a LaTeX compiler, so the
source was validated with static checks rather than a local PDF build.
