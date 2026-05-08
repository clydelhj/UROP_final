# Evaluation Script Planning for different method for unpaired translation of autofrozen images -> HE Images
## Evaluation Metrics
- Distribution-based (global quality)
    - Frechet Inception Distance (FID)
    - Frechet MedicalNet Distance(FMD): A medical-specific adapation of FID that uses features from a network pre-trained on medical data rather than natural images, making it more sensitive to histological features

- Perceptual metrics
    - NIQE (Natural Image Quality Evaluator): A Completely "blind" metric that measures deviations from a "natural" image model built from a corpus of high-quality iamges
    - LPIPS (Learned Perceptual Image Patch Similarity) 
    - Inception Score (IS): Measures the quality and diversity of the generated images based on how clearly a classifier can identify objects within them.
    - [Domain specific] PaPIS (Pathology-Aware Perceptual Image Similarity): A newer metric specifically tailored for histology that focuses on cellular morphology and tissue architecture rather than just textural fidelity
    - [Domain specific] HSFI (Histological Structural Fidelity Index): A novel metric designed to capture diagnostic details and structural alignment in virtual staining workflows
    - Other relevant metrics which could be relevant for Autofrozen -> HE untailored translation using various models: CUT, CycleGAN, Syndiff, Conditional Flow Matching.
    