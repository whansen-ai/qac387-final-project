# Build 0 – Data Analysis Pipeline (Assignment 1)

**Course:** QAC387  
**Assignment Type:** Individual  
**Focus:** Python functions, error checks, reproducible analysis pipelines  

---

## Purpose of the application 

The purpose of this application is to help get better insights into which stocks are attractive value stocks. The approach is adopted from the book Quantative Value which seeks to create a mechanical approach to value investing to reduce human error and bias. 

The theory behind value investing is that for the most part, the market efficiently prices all securities because all investors have accurate, symmetric information, and trading transactions have low costs. However, there are rare opportunities that present themselves, where the market will overreact either positively or negatively to market news. This provides careful investors the opportunity to purchase stocks below their intrinsic value, or their true value. Eventually, the market will price these securities accurately, but in the meantime the investor who spotted the initial mispricing will gain. 

In this project we seek to develop an application that will reduce the search time for undervalued securites. 

## Instruction for using the application

First, clone the repo locally. 

Second, create and activate a virtual working environment (.venv) and download the requirements.txt dependancies using this bash command: 

pip install -r requirements.txt

Third, use this bash command to run the application: 

python builds/build3_hitl_router_agent.py --data data/sixfirmlist.csv --report_dir reports --tags build3 --memory

Fourth, interact with the AI agent by using commands of your choice. Note that your choice should be prefaced with "ask" or "tool". 

Fifth, press (y/n) to approve code development or tool usage from the AI agent. 

## Cuations for using the app 

Code generation from the AI agent may be incorrect or have minor bugs, all code generation and tool usage should be reviewed by the user first. 
The current dataset is not entirely complete, currently it includes on 6 companies for testing purposes, however using WRDS we will be able to easily replicate our dataset with more recent financial data as well as a more robust list of stocks, ideally in the range of 500-2000 stocks. Further, it is of note that the agent to still sensitive to work choice, if you ask the agent to use a tool without naming some of the key words explicitly it may not recognize the tool. Lastly, the financial data included does not include financial data from months past January of 2026 (01/2026). 

