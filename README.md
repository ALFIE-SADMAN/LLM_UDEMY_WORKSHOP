# 🧠 LLM ENGINEERING

*A Comprehensive Curriculum of Practical Labs, Experiments, and Applied Projects in Large Language Model Engineering*

---

## 📘 Overview

This repository consolidates eight weeks of structured coursework and extended applied projects exploring the foundations and engineering practices behind modern Large Language Models (LLMs).
It serves both as a **learning curriculum** and a **portfolio of implementations**, demonstrating an end-to-end understanding of model pipelines, fine-tuning, inference, evaluation, and deployment.

The repository combines:

* **Weekly instructional modules** (Week 1–Week 8) introducing core LLM concepts.
* **Diagnostics utilities** for environment verification.
* **Community contributions** featuring autonomous-agent architectures, multimodal applications, data-generation utilities, and evaluation pipelines.

---

## 🎯 Learning Objectives

1. Develop a robust understanding of LLM architectures and transformer mechanics.
2. Acquire practical experience in model fine-tuning, prompt engineering, and inference optimization.
3. Build autonomous AI agents integrated with web data, voice, or image modalities.
4. Learn best practices in MLOps, environment management, and reproducible experimentation.
5. Explore open-source frameworks such as Hugging Face Transformers, LangChain, and OpenAI’s API ecosystem.

---

## 🤉 Repository Structure

```bash
llm_engineering/
│
├── diagnostics.py / diagnostics.ipynb     # Environment verification
├── environment.yml / requirements.txt     # Dependency management
├── SETUP-*.md / SETUP-*.pdf               # OS-specific setup guides
├── week1–week8/                           # Core learning modules
├── community-contributions/               # Extended applied projects
│   ├── dungeon_extraction_game/           # Multimodal story generation engine
│   ├── WebScraperApp/                     # LLM-driven web scraping automation
│   ├── fitness-nutrition-planner-agent/   # Personalized agent using embeddings
│   ├── bojan-playwright-scraper/          # Async scraping and summarization
│   ├── multi-agent_gui_with_gradio/       # Voice and text interface for agents
│   ├── LLaVA-For-Visually-Impared-People/ # Multimodal visual assistant
│   ├── openai-twenty-questions/           # Interactive reasoning game
│   ├── Market_Research_Agent.ipynb        # Domain-specific RAG pipeline
│   └── SyntheticDataGenerator_PT.ipynb    # Procedural data augmentation
└── LICENSE / README.md / images/
```

---

## ⚙️ Environment Setup

You can recreate the environment using either **Conda** or **pip**.

### **Conda**

```bash
conda env create -f environment.yml
conda activate llm-engineering
python -m ipykernel install --user --name llm-engineering --display-name "Python (llm-engineering)"
```

### **pip + venv**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Run `python diagnostics.py` or open `diagnostics.ipynb` to verify that all dependencies and APIs are functioning correctly.

---

## 📚 Week-by-Week Curriculum Summary

| Week       | Focus Area                                     | Core Learning & Implementation Highlights                                                                                                           |
| ---------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Week 1** | *Foundations of Prompt Engineering*            | Understanding tokenization, temperature, and context length. Implementing zero-shot vs few-shot prompting strategies.                               |
| **Week 2** | *Transformer Mechanics*                        | Detailed study of self-attention, positional encoding, and parameterization. Includes visualization notebooks and custom attention implementations. |
| **Week 3** | *Embeddings and Vector Databases*              | Creating and indexing text embeddings; semantic search using FAISS / Chroma; similarity scoring metrics.                                            |
| **Week 4** | *Retrieval-Augmented Generation (RAG)*         | Constructing hybrid retrieval pipelines, integrating context windows, and evaluating precision-recall trade-offs.                                   |
| **Week 5** | *Fine-Tuning and Parameter-Efficient Training* | LoRA, QLoRA, and prefix-tuning demonstrations with practical code; includes experiments on resource-limited hardware.                               |
| **Week 6** | *Evaluation and Hallucination Detection*       | Designing automatic metrics (BLEU, ROUGE, BERTScore) and human-in-the-loop validation frameworks.                                                   |
| **Week 7** | *LLMOps and Deployment Pipelines*              | Creating reproducible training scripts, versioning with MLflow / DVC, and setting up lightweight inference APIs.                                    |
| **Week 8** | *Multi-Agent and Generative Applications*      | Integrating LangChain Agents, Gradio interfaces, and real-time reasoning systems. Culminates in autonomous multi-agent demo applications.           |

Each week is accompanied by annotated Jupyter notebooks and Python modules containing explanatory comments and experimental outputs.

---

## 🧠 Key Skills & Technologies Gained

| Domain                   | Skills / Tools                                                             |
| ------------------------ | -------------------------------------------------------------------------- |
| **Programming & ML Ops** | Python 3, PyTorch, TensorFlow, NumPy, Pandas, MLflow, DVC                  |
| **LLM Engineering**      | Prompt Engineering, Transformers, Fine-Tuning (LoRA/QLoRA), Quantization   |
| **Retrieval & Search**   | FAISS, Chroma, Embeddings, Vector Search, Hybrid RAG Design                |
| **Frameworks & APIs**    | Hugging Face Transformers, LangChain, OpenAI API, Gradio                   |
| **Agentic AI**           | LangChain Agents, Voice/Text Interfaces, Memory Chains                     |
| **Software Engineering** | Modular OOP Design, Environment Control, Error Logging, Testing            |
| **Deployment & Ops**     | Git, Conda/pip, Docker (optional), Reproducibility, Continuous Integration |

---

## 🤉 Project Taxonomy: Community Contributions

### 🔹 **Dungeon Extraction Game**

A multimodal adventure generator combining GPT-based storytelling with DALLE/Gemini/Grok illustration modules.
It demonstrates **agent orchestration**, **image-text synergy**, and **tool-calling** for adaptive narrative generation.

**Key Concepts:**
LLM Agents · Function Calling · Multimodal Generative AI · Dynamic Workflow Pipelines

---

### 🔹 **Fitness & Nutrition Planner Agent**

A personal-assistant model that analyzes a user’s profile (JSON input) to generate tailored diet and workout plans.
Uses **LangChain**, **OpenAI functions**, and structured output parsing.

**Key Concepts:**
Agentic Reasoning · Prompt Templates · JSON Parsing · Personalized Recommendation

---

### 🔹 **Web Scraper App**

A fully modular LLM-enhanced web-scraping framework built with Playwright and Python’s `requests`, enabling both data extraction and content summarization.

**Key Concepts:**
Playwright Automation · Async I/O · Web Content Summarization · Data Pipeline Integration

---

### 🔹 **LLaVA for Visually Impaired People**

Adaptation of the LLaVA (Vision-Language Model) pipeline to generate real-time captions from camera input—aimed at accessibility research.

**Key Concepts:**
Multimodal Perception · Image Captioning · Accessibility AI · Real-Time Inference

---

### 🔹 **Multi-Agent GUI with Gradio**

A minimalistic multi-agent coordination demo using LangChain and Gradio UI, combining **voice input**, **text output**, and **memory persistence**.

**Key Concepts:**
Conversational UI · Gradio Deployment · LangChain Memory · Voice Integration

---

### 🔹 **Synthetic Data Generator (PT)**

A parameterized synthetic text generation utility for augmenting fine-tuning datasets, ensuring better class balance and reduced overfitting.

**Key Concepts:**
Data Augmentation · Prompt-based Generation · Controlled Sampling · Dataset Scaling

---

### 🔹 **Market Research Agent**

Automates comparative market analysis through retrieval-augmented queries and structured summarization, integrating OpenAI and Ollama back-ends.

**Key Concepts:**
Retrieval-Augmented Generation (RAG) · Knowledge Graphs · API Fusion

---

## 🚀 Running Labs and Projects

### Diagnostics

```bash
python diagnostics.py
# or
jupyter notebook diagnostics.ipynb
```

### Week Labs

```bash
cd week5
jupyter notebook fine_tuning_experiments.ipynb
```

### Example: Running an Agent

```bash
cd community-contributions/fitness-nutrition-planner-agent
python app.py
```

Outputs are logged under respective directories or notebooks.

---

## 💼 Recruiter-Oriented Highlights

* Demonstrates **hands-on experience with full-stack LLM workflows** — from data ingestion and embedding to inference and agentic deployment.
* Includes **production-ready modular code** (OOP architecture, logging, environment reproducibility).
* Reflects capability in **multimodal reasoning**, **retrieval-augmented generation**, and **autonomous agent design**.
* Portfolio aligns with modern AI-industry requirements: **LangChain**, **LLMOps**, **Generative AI APIs**, and **MLOps tools**.
* Provides clear academic depth suitable for research or teaching assistant roles in **AI Engineering**.

---

## 🧮 Evaluation Metrics & Research Context

| Evaluation Type        | Metrics Used                        | Purpose                      |
| ---------------------- | ----------------------------------- | ---------------------------- |
| **Text Generation**    | BLEU, ROUGE, METEOR                 | Quality & fluency            |
| **Semantic Retrieval** | Precision@k, Recall@k               | RAG effectiveness            |
| **Fine-Tuning**        | Perplexity & Loss                   | Convergence analysis         |
| **Agents**             | Response Latency, Task Success Rate | Practical deployment testing |

Academic reference frameworks include:

* Vaswani et al., *“Attention is All You Need,”* NeurIPS 2017.
* Raffel et al., *“Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,”* JMLR 2020 (T5).
* Hu et al., *“LoRA: Low-Rank Adaptation of Large Language Models,”* ICLR 2022.
* Schick & Schütze, *“Few-Shot Text Generation with Pattern-Exploiting Training,”* ACL 2021.

---

## 📊 Suggested Future Work

1. Integrate **OpenDevin**-style multi-agent coordination.
2. Add **evaluation dashboards** for RAG accuracy.
3. Extend **LLaVA** project with speech-to-caption pipeline.
4. Containerize entire environment with **Docker** for cloud deployment.

---

## 🗾 References

1. OpenAI API Documentation — *[https://platform.openai.com/docs](https://platform.openai.com/docs)*
2. Hugging Face Transformers — *[https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)*
3. LangChain Docs — *[https://python.langchain.com](https://python.langchain.com)*
4. Google AI Blog — *Attention Mechanisms and Transformer Research*
5. University of Adelaide AI/ML Curriculum References (2024–25)

---

## 🪪 License

This repository is distributed under the **MIT License**.
You are free to use, modify, and distribute this material provided attribution to the original author (`@ALFIE-SADMAN`) is maintained.

---

## 🏁 Author & Acknowledgements

Developed and maintained by **Sadman Alfie**,
*Master of Artificial Intelligence & Machine Learning, University of Adelaide*

Special thanks to the open-source contributors whose work underpins these educational projects.

