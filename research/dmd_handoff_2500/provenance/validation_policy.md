# Official validation and resource rules

The course source is the official FastGen `README.md` at commit `e3f5120000cd91037017d86f23993492195101a5`, copied verbatim to `source_overlay/README.md` and verified against git HEAD.

Its validation rule reads:

> **Do NOT train on the validation split:** Only the images listed in `train_split.txt` may be used for training, distillation, or hyper-parameter selection. The validation split is the FID reference set.

Training and distillation use of validation are prohibited; hyperparameter selection use is explicitly prohibited. The quoted sentence does not separately enumerate checkpoint/model selection. Using the development reference for that choice is the documented experiment policy. Validation is explicitly the FID reference for evaluation/reporting. No inspected course wording says merely viewing a score is prohibited. Historical override files add operational interpretations and must not be substituted for the official wording.

This handoff selects the 2500-step checkpoint on development FID. It records official metrics separately with their evidence status, and does not claim the entire historical workflow never inspected validation.

The VRAM rule reads:

> **Do NOT exceed 20 GB of peak training VRAM:** Each model must train within a single 20 GB GPU, matching the NVIDIA A100 vGPU provided through KCLOUD for this course. Measure peak memory (`torch.cuda.max_memory_allocated()`) and keep it below the limit.

The run asserts `<20_000_000_000` bytes; logs record 3,691,061,760 warmup and 4,522,762,752 main peak allocated bytes. These are historical measurements rechecked for consistency, not a new training-memory benchmark.

The official per-model parameter budget is 100,000,000, including registered frozen parameters and excluding buffers. The package's single generator has 48,883,107. Some inherited project documents use a stricter 60M target. Both are satisfied, but 100M is the quoted official rule. Inference NFE counts trained-network executions; the packaged generator has one backbone call and no auxiliary learned inference network.
