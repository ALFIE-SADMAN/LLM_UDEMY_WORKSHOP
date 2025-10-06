# 🤖 LLM Engineering Course

> **A comprehensive, hands-on journey from LLM fundamentals to production-ready AI systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991.svg)](https://platform.openai.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg)](https://jupyter.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Course Structure](#-course-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Detailed Setup](#-detailed-setup)
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [Learning Outcomes](#-learning-outcomes)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Resources](#-resources)

---

## 🎯 Overview

This **8-week intensive course** takes you from zero to hero in Large Language Model engineering. You'll build real-world applications, master cutting-edge techniques, and develop production-ready AI systems.

### What You'll Build

- 🌐 **AI-Powered Web Summarizer** - Extract and summarize web content using GPT
- 💬 **Intelligent Chatbots** - Multi-turn conversations with context awareness
- 🔍 **Semantic Search Engine** - Vector-based knowledge retrieval systems
- 🤖 **Autonomous AI Agents** - Multi-agent systems that collaborate to solve complex problems
- 🎯 **Custom Fine-Tuned Models** - Domain-specific LLMs tailored to your needs

### Who This Is For

- 🎓 **Developers** looking to integrate AI into their applications
- 📊 **Data Scientists** expanding into LLM engineering
- 💼 **Product Managers** wanting hands-on AI experience
- 🚀 **Entrepreneurs** building AI-powered products
- 🔬 **Researchers** exploring practical LLM applications

---

## 📚 Course Structure

### Week 1: Foundations
**Getting Started with LLMs**
- Python environment setup and Jupyter notebooks
- First API calls to OpenAI GPT models
- Building a web content summarizer
- Understanding prompts and responses

**Key Projects:** Web Scraper & Summarizer

---

### Week 2: Prompt Engineering Fundamentals
**Mastering Communication with AI**
- System vs user prompts
- Temperature, top_p, and sampling parameters
- Few-shot learning and prompt templates
- Handling API responses and error management

**Key Projects:** Custom Prompt Templates, Multi-Model Comparison Tool

---

### Week 3: Advanced Prompting Techniques
**Optimizing LLM Behavior**
- Chain-of-thought reasoning
- Role-based prompting strategies
- Prompt iteration and A/B testing
- Context window optimization

**Key Projects:** Advanced Reasoning System, Prompt Optimizer

---

### Week 4: Embeddings & Optimization
**Understanding Semantic Representations**
- Text embeddings and vector spaces
- Cosine similarity and distance metrics
- Performance optimization techniques
- Caching and rate limiting strategies

**Key Projects:** Semantic Similarity Engine, Performance Dashboard

---

### Week 5: Vector Databases & Knowledge Bases
**Building Intelligent Search Systems**
- ChromaDB and vector storage
- Semantic search implementation
- Building and querying knowledge bases
- Hybrid search strategies

**Key Projects:** Personal Knowledge Base, Semantic Search API

---

### Week 6: RAG Systems
**Retrieval-Augmented Generation**
- RAG architecture and design patterns
- Document chunking strategies
- Context injection and prompt engineering
- Evaluation metrics and quality assessment

**Key Projects:** Document Q&A System, RAG Evaluation Framework

---

### Week 7: Fine-Tuning Models
**Customizing LLMs for Your Domain**
- Dataset preparation and formatting
- Fine-tuning with OpenAI API
- Model evaluation and comparison
- Deployment strategies

**Key Projects:** Custom Domain Expert Model, Model Performance Analyzer

---

### Week 8: AI Agents & Agentic Systems
**Building Autonomous AI**
- Agent architectures and frameworks
- Tool use and function calling
- Multi-agent collaboration
- LangChain and agent orchestration

**Key Projects:** Multi-Agent Research Assistant, Autonomous Task Solver

---

## ✅ Prerequisites

### Required
- **Python Knowledge**: Basic understanding (variables, functions, loops)
- **Command Line**: Familiarity with terminal/command prompt
- **Text Editor**: VS Code, PyCharm, or similar
- **OpenAI API Key**: [Sign up here](https://platform.openai.com/signup)

### Recommended
- Git version control basics
- Understanding of APIs and HTTP requests
- Experience with Jupyter notebooks
- Basic understanding of machine learning concepts

### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **Internet**: Stable connection for API calls

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm_engineering.git
cd llm_engineering
```

### 2. Create Environment (Conda - Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate llms
```

**OR** using pip/virtualenv:

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-proj-your-api-key-here
```

> 🔑 **Get your API key**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 4. Verify Setup

```bash
python diagnostics.py
```

This will check your environment, API key, and dependencies.

### 5. Launch JupyterLab

```bash
jupyter lab
```

Navigate to `week1/day1.ipynb` and start learning! 🎉

---

## 🛠️ Detailed Setup

### Platform-Specific Guides

- 🪟 **[Windows Setup Guide](SETUP-PC.md)** ([PDF](SETUP-PC.pdf))
- 🍎 **[macOS Setup Guide](SETUP-mac.md)** ([PDF](SETUP-mac.pdf))
- 🐧 **[Linux Setup Guide](SETUP-linux.md)** ([PDF](SETUP-linux.pdf))

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=sk-proj-xxxxx

# Optional (for advanced weeks)
ANTHROPIC_API_KEY=sk-ant-xxxxx
GOOGLE_API_KEY=xxxxx
WANDB_API_KEY=xxxxx
```

### Installing Additional Dependencies

For specific weeks, you may need additional packages:

```bash
# Week 5-6: Vector databases
pip install chromadb sentence-transformers

# Week 7: Fine-tuning
pip install datasets evaluate

# Week 8: Agent frameworks
pip install langchain langchain-openai
```

---

## 🔧 Technologies

### Core LLM Providers
| Technology | Purpose | Week Introduced |
|------------|---------|-----------------|
| **OpenAI GPT** | Primary LLM for chat & completion | Week 1 |
| **Anthropic Claude** | Alternative LLM provider | Week 2 |
| **Google Gemini** | Multi-modal AI capabilities | Week 3 |

### Frameworks & Libraries
| Technology | Purpose | Week Introduced |
|------------|---------|-----------------|
| **LangChain** | LLM application framework | Week 8 |
| **ChromaDB** | Vector database for embeddings | Week 5 |
| **Sentence Transformers** | Text embedding models | Week 4 |
| **BeautifulSoup** | Web scraping | Week 1 |
| **Gradio** | Quick ML model UIs | Week 6 |

### Development Tools
| Tool | Purpose |
|------|---------|
| **JupyterLab** | Interactive development environment |
| **Weights & Biases** | Experiment tracking and logging |
| **Python dotenv** | Environment variable management |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib/Plotly** | Data visualization |

---

## 📁 Project Structure

```
llm_engineering/
│
├── 📂 week1/                        # Foundations
│   ├── day1.ipynb                   # First LLM API calls
│   ├── day2 EXERCISE.ipynb          # Practice exercises
│   ├── day5.ipynb                   # Week 1 project
│   ├── Guide to Jupyter.ipynb       # Jupyter tutorial
│   ├── Intermediate Python.ipynb    # Python refresher
│   └── troubleshooting.ipynb        # Common issues
│
├── 📂 week2/                        # Prompt Engineering
│   ├── day1.ipynb → day5.ipynb      # Daily lessons
│   └── community-contributions/     # Student work
│
├── 📂 week3/                        # Advanced Prompting
├── 📂 week4/                        # Embeddings & Optimization
├── 📂 week5/                        # Vector Databases
├── 📂 week6/                        # RAG Systems
├── 📂 week7/                        # Fine-Tuning
├── 📂 week8/                        # AI Agents
│
├── 📄 diagnostics.py                # Environment checker
├── 📄 requirements.txt              # Python dependencies
├── 📄 environment.yml               # Conda environment
├── 📄 .env.example                  # Environment template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 LICENSE                       # License information
└── 📄 README.md                     # This file
```

---

## 🎓 Learning Outcomes

By the end of this course, you will be able to:

### Technical Skills
✅ Make API calls to multiple LLM providers (OpenAI, Anthropic, Google)
✅ Design effective prompts for different use cases
✅ Implement semantic search using vector embeddings
✅ Build RAG systems for question-answering over custom documents
✅ Fine-tune models for domain-specific tasks
✅ Create multi-agent systems that collaborate autonomously
✅ Deploy LLM applications to production

### Practical Knowledge
✅ Optimize API costs and performance
✅ Handle rate limits and error scenarios
✅ Evaluate and compare model outputs
✅ Build scalable LLM architectures
✅ Implement security best practices for API keys
✅ Debug and troubleshoot LLM applications

### Business Applications
✅ Summarization systems for content processing
✅ Intelligent chatbots and virtual assistants
✅ Knowledge base Q&A systems
✅ Automated content generation
✅ Document analysis and extraction
✅ Research and analysis automation

---

## 🔍 Troubleshooting

### Running Diagnostics

```bash
python diagnostics.py
```

This checks:
- ✓ Python version and environment
- ✓ Required packages and versions
- ✓ API key configuration
- ✓ Network connectivity
- ✓ System resources (RAM, disk)

Output is saved to `report.txt` for sharing.

### Common Issues

#### 🔴 API Key Not Found
**Error:** `No API key was found`

**Solution:**
1. Ensure `.env` file exists in project root
2. Check the format: `OPENAI_API_KEY=sk-proj-...`
3. No quotes needed around the key
4. No spaces before or after `=`

#### 🔴 Import Errors
**Error:** `ModuleNotFoundError: No module named 'openai'`

**Solution:**
```bash
# Ensure environment is activated
conda activate llms  # or: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 🔴 Jupyter Kernel Issues
**Error:** `Kernel not found` or wrong Python version

**Solution:**
```bash
# Install Jupyter kernel for your environment
python -m ipykernel install --user --name=llms
```

#### 🔴 Rate Limit Errors
**Error:** `Rate limit exceeded`

**Solution:**
- Add delays between API calls: `time.sleep(1)`
- Implement exponential backoff
- Check your OpenAI usage limits
- Consider upgrading your API plan

### Additional Help

1. 📖 Check [week1/troubleshooting.ipynb](week1/troubleshooting.ipynb)
2. 🔍 Review platform-specific setup guides
3. 📧 Contact: ed@edwarddonner.com
4. 🐛 Open an issue on GitHub

---

## 🤝 Contributing

We welcome contributions from the community!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Ideas

- 🐛 Fix bugs or typos
- 📝 Improve documentation
- ✨ Add new examples or exercises
- 🎨 Create visualizations or diagrams
- 🔧 Optimize existing code
- 🌐 Add alternative implementations (e.g., Selenium web scraper)

### Community Contributions

Each week folder includes a `community-contributions/` directory with examples and extensions from students. Feel free to explore and add your own!

---

## 📖 Resources

### Official Documentation
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Course Materials
- [Jupyter Notebook Guide](week1/Guide%20to%20Jupyter.ipynb)
- [Python Refresher](week1/Intermediate%20Python.ipynb)
- [Troubleshooting Guide](week1/troubleshooting.ipynb)

### Additional Learning
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)

### Community & Support
- 💬 Course Discussion Forum (coming soon)
- 📧 Email: ed@edwarddonner.com
- 💼 LinkedIn: [linkedin.com/in/eddonner](https://linkedin.com/in/eddonner)
- 🐦 Twitter: [@edwarddonner](https://x.com/edwarddonner)

---

## 📊 API Cost Estimates

### Expected Usage Per Week
| Week | Estimated Cost | Notes |
|------|----------------|-------|
| 1-2  | $1-3 | Basic API calls, small prompts |
| 3-4  | $3-5 | More experimentation, embeddings |
| 5-6  | $5-10 | RAG systems, larger context |
| 7-8  | $10-20 | Fine-tuning, agent iterations |

**Total Course Estimate:** $25-50 USD

💡 **Cost Saving Tips:**
- Use `gpt-4o-mini` for development (cheaper than GPT-4)
- Cache responses when possible
- Implement rate limiting
- Monitor usage on [platform.openai.com/usage](https://platform.openai.com/usage)

---

## 🔐 Security Best Practices

### API Key Safety
- ✅ Never commit `.env` files to Git
- ✅ Use environment variables, not hardcoded keys
- ✅ Rotate keys regularly
- ✅ Set spending limits in OpenAI dashboard
- ✅ Use read-only keys when possible

### .gitignore Example
```gitignore
.env
*.pyc
__pycache__/
.ipynb_checkpoints/
venv/
*.log
report.txt
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Course Instructor
**Ed Donner** - LLM Engineering Expert
📧 ed@edwarddonner.com
🔗 [linkedin.com/in/eddonner](https://linkedin.com/in/eddonner)

### Contributors
Special thanks to all students who have contributed examples, bug fixes, and improvements. See individual `community-contributions/` folders for attribution.

### Technologies
Built with ❤️ using:
- OpenAI GPT Models
- Python & Jupyter
- ChromaDB, LangChain
- And many other amazing open-source projects

---

## 🚦 Getting Started Checklist

- [ ] Python 3.10+ installed
- [ ] Git installed and configured
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with API key
- [ ] Diagnostics script passed (`python diagnostics.py`)
- [ ] JupyterLab launches successfully
- [ ] `week1/day1.ipynb` runs without errors

**Ready to start?** Navigate to [week1/day1.ipynb](week1/day1.ipynb) and begin your LLM journey! 🚀

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for aspiring LLM Engineers

[Report Bug](https://github.com/yourusername/llm_engineering/issues) · [Request Feature](https://github.com/yourusername/llm_engineering/issues) · [Ask Question](https://github.com/yourusername/llm_engineering/discussions)

</div>