Our programming is in Python, and here is a note on all the data files and code files:

[1] File "PDT Database Original", "PCR Database Original", and "PMR Database Original" are the zircon databases we have compiled and you can find information of sampling location and references. "PDT Database Original" contains 9649 data, "PCR Database Original" contains 6001 data, and "PMR Database Original" contains 1598 data. 

[2] File "PDT_Database", "PCR_Database" or "PMR_Database" only retain data and information of features and labels, which we use to train classification models for porphy deposit types (PDT), porphyry Cu reserves (PCR), and porphyry Mo reserves (PMR).

[3] File "WS23-3-1", "WS23-3-6", "WS23-3-11", "WS23-4-5", "WS23-4-11", and "WS23-6-1" contain zircon composition data from Wunugetushan deposit, which we used for the case study.  

[4] File "main_PDT", "main_PCR", and "main_PMR" are Python files, which are code for models PDT, PCR, and PMR respectively.

[5] File "apply" is a Python file, which is the code for the case study and application. Note that when you use this code to run your own data files, you need to replace the "PDT_Database", "PCR_Database" or "PMR_Database" file in the code depending on your purpose and input your filename.


If you are interested in our code and research, please refer to paper Zi-Hao Wen, Bo Xu, Christopher Kirkland, David Lentz, Zeng-Qian Hou and Tao Wang, (2024), A Machine Learning Zircon Trace Element Tool to Predict Porphyry Deposit Type and Resource Size, and feel free to cite it.
