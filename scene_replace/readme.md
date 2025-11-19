## <a id="top"></a>Quick Start for Scene Replacement

<p align="left">
  <a href="#introduction"><b>Introduction</b></a> |
<a href="#preparation"><b>Preparation</b></a> |
  <a href="#quickstart"><b>Quick Start</b></a>


[//]: # (  <a href="#extract"><b>Compute Accuracy</b></a> |)
</p>

### <a id="introduction"></a>Introduction

---

This project focuses on **scene replacement** in USD-based environments. The goal is to enrich and augment 3D scenes for both **sim2real applications** and **virtual scene enhancement**.

* **Sim2Real**: By scanning real-world layouts and applying the scene replacement pipeline, we can generate high-quality virtual scenes that preserve the original spatial arrangement.
* **Virtual Scene Augmentation**: For existing InternUtopia scenes, we enhance diversity by performing random replacements based on similarity retrieval.

The scene replacement process consists of four main steps:

1. **Object Retrieval** – Identify replaceable 3D objects within the target USD scene.
2. **Asset Metadata Generation** – Scan the InternUtopia 3D asset library and generate metadata for each object.
3. **Similarity-Based Matching** – For each target object, search the asset library to find semantically or structurally similar replacements.
4. **Replacement & Validation** – Execute the replacement while preserving the original object’s transforms, followed by collision detection to ensure physical plausibility.

This pipeline enables scalable scene diversification, supporting both realistic simulation and richer virtual environments.

### <a id="Preparation"></a>Preparation

---

First, you need to download the **InternUtopia** asset library.  
If you have not downloaded it yet, please refer to [InternUtopia](https://github.com/InternRobotics/InternUtopia) for instructions on dataset preparation.  

### <a id="quickstart"></a>Quick Start

---

#### Step 0: Install requirements
```bash
conda create --name scene_rep python=3.10 -y
conda activate scene_rep
pip install -r requirements.txt
```

#### Step 1: Generate Meta-Information for 3D Objects in InternUtopia

In this step, we generate **metadata** for each 3D object in the InternUtopia asset library.
The metadata mainly includes:

* **Object dimensions** (currently the most important feature in this project, used for retrieving size-proportional objects)
* **Material-related attributes** (reserved for future extensions)

Run the following command to generate metadata:

```bash
python generate_meta_info_utopia.py --scene_root </path/to/internutopia/assets/scenes/GRScenes-100/home_scenes>
```

#### Step 2: Extract Replaceable Objects and Perform Similarity-Based Retrieval

This step is divided into two sub-processes:

1. **Extract replaceable objects from the original USD scene**.
2. **Similarity-based retrieval**:

   * For each object category in the scene, generate **word embeddings** using `gensim`.
   * Generate embeddings for all categories in the InternUtopia asset library.
   * Perform similarity matching to find the most relevant categories.
   * Within the matched category, select candidate objects with **size compatibility** using the constraint:
    `min(ori_size, new_size) / max(ori_size, new_size) > 0.75`
   * Construct a mapping dictionary from old objects to their replacement objects.

Run the following command:

```bash
python extract_replace_obj.py --usd_path </path/to/replace/usd/file/bounding_box_scene.usda>
```

#### Step 3: Replace the 3D Objects  

Based on the matching dictionary generated in **Step 2**, the corresponding 3D objects are replaced in the scene.  
- The replacement preserves the original object’s **transformations** (position, rotation, and scale).  
- After replacement, a **collision detection** step is executed to ensure the physical plausibility of the scene.  

Run the following command:  
```bash
python replace.py --usd_path </path/to/replace/usd/file/bounding_box_scene.usda> --scene_root </path/to/internutopia/assets/scenes/GRScenes-100/home_scenes> --save_usd_path </path/to/save/new_scene.usda>
```

#### 🔝 [Back to Top](#top)