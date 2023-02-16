# Global-inhibition-in-head-direction-neural-circuits
Code data for published paper:  https://doi.org/10.1007/s00359-023-01615-z 

Code for the main five experiment protocols
1. [Robustness and Motion Persistency Test](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/tree/main/robustness)
2. [Static Persistency Test](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/tree/main/static_persistency)
3. [Speed Test](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/tree/main/speed)
4. [Dynamic Characteristic Test](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/tree/main/dynamic_characteristic)

### Robustness and Motion Persistency Test

- **cx_model_table_E16.xls**: Connectome data for E16 model  
- **cx_model_table_E18.xls**: Connectome data for E18 model    
- **flysim07_21_4_macOS.out**: Flysim model for mac users  
- **flysim07_21_4.out**: Flysim model for linux/windows users  
- **flysimpler.py**: Functions and classes for the model  
- **gen_conf_pro.py**: Main file to construct the network, generates .conf and .pro files for the flysim model  
      - To modified the synaptic weight bases (K), go to [line 50-57](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/robustness/gen_conf_pro.py#L50-L57)  
      - Simply sets the synaptic weight base to zero for switching models, ex: Delta7_EPG = EPG_Delta7 = 0 for R models  
      - Switch E16 model to E18 model by reading different files in [line 72](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/robustness/gen_conf_pro.py#L72)
- **motion_persistency.py**: Scoring the performance for the motion persistency test  
      - Evaluates the score from output gau.dat files, fill in the output filename in [line 141](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/robustness/motion_persistency.py#L141)
- **plot.py**: Ploting the figures  
- **run.sh**: Runs the whole program, remember to switch the model according to your OS  

### Static Persistnecy Test

- **cx_model_table_E16.xls**: Connectome data for E16 model  
- **cx_model_table_E18.xls**: Connectome data for E18 model    
- **flysim07_21_4_macOS.out**: Flysim model for mac users  
- **flysim07_21_4.out**: Flysim model for linux/windows users  
- **flysimpler.py**: Functions and classes for the model  
- **gen_conf_pro.py**: Main file to construct the network, generates .conf and .pro files for the flysim model  
      - To modified the synaptic weight bases (K), go to [line 50-57](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/static_persistency/gen_conf_pro.py#L50-L57)  
      - Simply sets the synaptic weight base to zero for switching models, ex: Delta7_EPG = EPG_Delta7 = 0 for R models  
      - Switch E16 model to E18 model by reading different files in [line 72](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/static_persistency/gen_conf_pro.py#L72)
- **plot.py**: Ploting the figures  
- **run.sh**: Runs the whole program, remember to switch the model according to your OS  
- **std.py**: Scoring the performance for the static persistency test  
      - Evaluates the score from output gau.dat files, fill in the output filename in [line 47](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/static_persistency/std.py#L47)

### Speed Test

- **classify.py**: Classify the bump type  
      - Classify bump type from output gau.dat files, fill in the output filename in [line 41](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/speed/classify.py#L41)  
- **cx_model_table_E16.xls**: Connectome data for E16 model  
- **cx_model_table_E18.xls**: Connectome data for E18 model    
- **flysim07_21_4_macOS.out**: Flysim model for mac users  
- **flysim07_21_4.out**: Flysim model for linux/windows users  
- **flysimpler.py**: Functions and classes for the model  
- **gen_conf_pro.py**: Main file to construct the network, generates .conf and .pro files for the flysim model  
      - To modified the synaptic weight bases (K), go to [line 50-57](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/speed/gen_conf_pro.py#L50-L57)  
      - Simply sets the synaptic weight base to zero for switching models, ex: Delta7_EPG = EPG_Delta7 = 0 for R models  
      - Switch E16 model to E18 model by reading different files in [line 72](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/speed/gen_conf_pro.py#L72)  
      - Change visual cue shift speed in [line 107](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/speed/gen_conf_pro.py#L107), (from 50~)   
- **plot.py**: Ploting the figures  
- **run.sh**: Runs the whole program, remember to switch the model according to your OS  

### Dynamic Characteristic

- **classify.py**: Classify the bump type  
      - Classify bump type from output gau.dat files, fill in the output filename in [line 50](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/dynamic_characteristic/classify.py#L50)  
- **cx_model_table_E16.xls**: Connectome data for E16 model  
- **cx_model_table_E18.xls**: Connectome data for E18 model    
- **flysim07_21_4_macOS.out**: Flysim model for mac users  
- **flysim07_21_4.out**: Flysim model for linux/windows users  
- **flysimpler.py**: Functions and classes for the model  
- **gen_conf_pro.py**: Main file to construct the network, generates .conf and .pro files for the flysim model  
      - To modified the synaptic weight bases (K), go to [line 50-57](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/dynamic_characteristic/gen_conf_pro.py#L50-L57)  
      - Simply sets the synaptic weight base to zero for switching models, ex: Delta7_EPG = EPG_Delta7 = 0 for R models  
      - Switch E16 model to E18 model by reading different files in [line 72](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/dynamic_characteristic/gen_conf_pro.py#L72)  
      - Change angle between visual cues (45˚, 90˚ , 135˚ or 180˚) in [line 107](https://github.com/CCLoLab/Global-inhibition-in-head-direction-neural-circuits/blob/f59d2d407a4f2d98d087ca46157aecc5a51a3efc/dynamic_characteristic/gen_conf_pro.py#L107)     
- **plot.py**: Ploting the figures  
- **run.sh**: Runs the whole program, remember to switch the model according to your OS



