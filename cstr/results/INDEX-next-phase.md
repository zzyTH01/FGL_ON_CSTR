# Next-phase result index

## transfer/transfer_teacher_lopo.csv
- **Experiment**: `cstr/run.py -e transfer_generalization`
- **Question**: cross-τ generalization of pretrained, finetuned, and scratch teachers
- **Unit**: one row per target dataset × condition × seed
- **Fields**: `path`, `condition`, `teacher_mse`
- **Reading**: compare `zero_shot` vs `finetuned` vs `scratch` per `path`; lower is better.

## teacher_transfer/teacher_optimization_transfer.csv
- **Experiment**: `cstr/run.py -e teacher_student_relation`
- **Question**: how much teacher-side MSE improvement transfers to the distilled student
- **Unit**: one paired row per condition × seed
- **Fields**: raw MSE, `teacher_gain`, `student_gain`, `transfer_ratio`, relative gains, and `normalized_transfer`
- **Reading**: positive gains mean improvement; use aggregated bootstrap CIs, not isolated ratios.

## complex/complex_cstr_fgl.csv
- **Experiment**: `cstr/run.py -e complex_multichannel`
- **Question**: whether complex multichannel CSTR outputs change FGL effectiveness
- **Unit**: one row per channel × L/H × seed
- **Fields**: channel metadata and standard `baseline_mse`, `teacher_mse`, `student_mse`, `fgl_delta`
- **Reading**: positive `fgl_delta` means FGL improved over the no-distillation baseline.
