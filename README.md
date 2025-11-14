# mAgent2 – AGI SDK Agent

This repository contains `mAgent2.py`, a custom agent designed to run with the [AGI SDK](https://github.com/agi-inc/agi-sdk).

On the web agent eval benchmark [REAL v1](https://realevals.xyz/), mAgent2 scores first on all three leaderboards. mAgent2 was designed before the release of REAL v2.

## 🚀 Prerequisites
Before using this agent, make sure you have:
- Python 3.9+ installed  
- Git installed  
- The **AGI SDK** cloned locally  

Follow the official [AGI SDK Local Development Guide](https://github.com/agi-inc/agi-sdk) to install and set up the SDK on your machine.

Check out the [writeup](https://docs.google.com/document/d/1Xb9fG6uSjcdcD3AjywBPJCE6xiYkzNZ6SjY2ln36GjE/edit?usp=sharing)!

Check out the [demo](https://drive.google.com/file/d/1jR3-dGKhwcZsNaWtf_TamlhC6JKbZHrx/view?usp=sharing)!

---
## 📊 Agent Evaluation Results

The `mAgent2` agent was evaluated on multiple tasks with the following results:

| Task Type   | Success | Total |
|-------------|---------|-------|
| Dashdish    | 9       | 11    |
| Fly-unified | 9       | 14    |
| Gocalendar  | 3       | 10    |
| Gomail      | 5       | 8     |
| Networkin   | 7       | 10    |
| Omnizon     | 9       | 10    |
| Opendining  | 4       | 10    |
| Staynb      | 5       | 9     |
| Topwork     | 1       | 9     |
| Udriver     | 8       | 11    |
| Zilloft     | 1       | 10    |

**Total Successes:**  
9 + 9 + 3 + 5 + 7 + 9 + 4 + 5 + 1 + 8 + 1 = 61

**Overall Performance:**  
61 / 112 = 0.5446 (≈ 54.5%)

---

## 📦 Setup Instructions

1. **Clone the AGI SDK repository** (if not already done):  
   ```bash
   # Clone the repository
    git clone https://github.com/agi-inc/agisdk.git
    cd agisdk

    # Install in development mode
    pip install -e .
    ```
2. **Move mAgent2.py into the AGI SDK folder:**
Copy the agent file into the SDK’s main folder (where the core modules live):
   ```bash
   mv /path/to/mAgent2.py ./agisdk/
   ```
---

## ▶️ Running the Agent

The following command will run the agent with Claude Opus 4.1 on the Omnizon-1 task:  
   ```bash
   python mAgent2.py --model claude-opus-4-1-20250805 --task webclones.omnizon-1 --headless=False
   ```
Please refer to the AGI SDK documentation for additional run parameters.
