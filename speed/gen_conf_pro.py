from flysimpler import Network
import pandas as pd
import sys


def add_connection(net, connection_table: pd.DataFrame, syn_info, weight=1):
    for row in connection_table.itertuples():
        for col in connection_table.columns:
            val = getattr(row, col)
            if val:
                net.add_target(row.Index, col, syn_info["neurotransmitter"], syn_info["meaneff"], val)


# Generate Network ---------------------------------------
net = Network()


# Generate Conf File -------------------------------------

# (Name, Tau, RevPot, FreqExt, MeanExtEff, MeanExtCon)
net.add_receptor("Ach", 20, 0, 0, 2.1, 1)
net.add_receptor("NMDA", 100, 0, 0, 2.1, 1)
net.add_receptor("GABA", 5, -70, 0, 2.1, 1)


# (Name, N, C, Taum, RestPot, ResetPot, Threshold)
EPG_num = 18
PEN_num = 16
net.add_neuron_population("EPG", EPG_num, group=True)
net.add_neuron_population("PEN", PEN_num, group=True)

net.add_neuron("RingEPG")
net.add_neuron("Half_PB_R")
net.add_neuron("Half_PB_L")

Delta7_postfix = ["L9L1R8", "L8R1R9", "L7R2", "L6R3", "L5R4", "L4R5", "L3R6", "L2R7"]
for postfix in Delta7_postfix:
    net.add_neuron(f"Delta7_{postfix}")

net.set_neuron_param_all(3, 0.1, 15, -70, -70, -50)
net.set_neuron_param("Half_PB_L", "N", 50)  # Half_PB_L.N = 50
net.set_neuron_param("Half_PB_R", "N", 50)  # Half_PB_R.N = 50

net.set_neuron_receptor_all("GABA", "NMDA", "Ach")


# (Name, TargetReceptor, MeanEff, weight)

# MeanEff
EPG_to_EPG = 1
EPG_to_PEN = 12.2
PEN_to_EPG = 13.6
EPG_to_Reip = 7
Reip_to_EPG = 7
Half_to_PEN =0.3
EPG_Delta7 = 7
Delta7_EPG = 7


syn = {"EPG_EPG": {"neurotransmitter": "NMDA", "meaneff": EPG_to_EPG},
       "EPG_PEN": {"neurotransmitter": "NMDA", "meaneff": EPG_to_PEN},
       "PEN_EPG": {"neurotransmitter": "NMDA", "meaneff": PEN_to_EPG},
       "EPG_RingEPG": {"neurotransmitter": "NMDA", "meaneff": EPG_to_Reip},
       "RingEPG_EPG": {"neurotransmitter": "GABA", "meaneff": Reip_to_EPG},
       "EPG_Delta7_pattern1": {"neurotransmitter": "NMDA", "meaneff": EPG_Delta7},
       "EPG_Delta7_pattern2": {"neurotransmitter": "NMDA", "meaneff": EPG_Delta7},
       "Delta7_EPG": {"neurotransmitter": "GABA", "meaneff": Delta7_EPG},
       "HalfPB_PEN": {"neurotransmitter": "NMDA", "meaneff": Half_to_PEN},
}

# Read E16 / E18 connectome
xls = pd.ExcelFile("cx_model_table_E16.xls")
base_sheets = ["EPG_PEN", "PEN_EPG", "HalfPB_PEN", "EPG_RingEPG", "RingEPG_EPG", "EPG_Delta7_pattern2", "Delta7_EPG", "EPG_EPG"]


for table in base_sheets:
    df = pd.read_excel(xls, table, index_col=0)
    add_connection(net, df, syn[table], 1)


# (GroupName, member)

# _macro_0: L0, R8
net.add_group("_macro_0", "PEN0")
# _macro_1: L1, R7
net.add_group("_macro_1", "PEN1", "PEN8")
# _macro_2: L2, R6
net.add_group("_macro_2", "PEN2", "PEN9")
# _macro_3: L3, R5
net.add_group("_macro_3", "PEN3", "PEN10")
# _macro_4: L4, R4
net.add_group("_macro_4", "PEN4", "PEN11")
# _macro_5: L5, R3
net.add_group("_macro_5", "PEN5", "PEN12")
# _macro_6: L6, R2
net.add_group("_macro_6", "PEN6", "PEN13")
# _macro_7: L7, R1
net.add_group("_macro_7", "PEN7", "PEN14")
# _macro_8: L8, R0
net.add_group("_macro_8", "PEN15")


# Generate Pro File -------------------------------------

VISUAL_FR = 50
PB_HALF_FR = 70
SPEED = 500


def shift_bump(net, interval=2000, VISUAL_FR=50, round=1):
    back_time = 8 * interval
    trial_time = 16 * interval
    for rnd in range(round):
        for i in range(8):
            net.add_event(rnd * trial_time + interval * i, 'ChangeExtFreq', 'PEN', 'Ach', 0)
            if i == 0:
                net.add_event(rnd * trial_time + interval * i + 1, 'ChangeExtFreq', '_macro_0', 'Ach', VISUAL_FR)
                net.add_event(rnd * trial_time + interval * i + 1, 'ChangeExtFreq', '_macro_8', 'Ach', VISUAL_FR)
            else:
                net.add_event(rnd * trial_time + interval * i + 1, 'ChangeExtFreq', '_macro_' + str(i), 'Ach', VISUAL_FR)

        for i in range(8):
            net.add_event(rnd * trial_time + back_time + interval * i, 'ChangeExtFreq', 'PEN', 'Ach', 0)
            if i == 7:
                net.add_event(rnd * trial_time + back_time + interval * i + 1, 'ChangeExtFreq', '_macro_0', 'Ach', VISUAL_FR)
                net.add_event(rnd * trial_time + back_time + interval * i + 1, 'ChangeExtFreq', '_macro_8', 'Ach', VISUAL_FR)
            else:
                net.add_event(rnd * trial_time + back_time + interval * i + 1, 'ChangeExtFreq', '_macro_' + str(7-i), 'Ach', VISUAL_FR)

    net.add_event(round * trial_time + 1, 'EndTrial')

ROUND = 1
if SPEED==50:
    ROUND = 4
elif SPEED == 100 or SPEED == 150:
    ROUND = 2
elif SPEED==25:
    ROUND=8

shift_bump(net, interval=SPEED, round=ROUND)




# (File_Name, Output_Type, Population, *args)
net.add_output("FiringRateALL.dat", "F", "AllPopulation", 100, 10)


# Output -------------------------------------------------

net.output("network.conf", "network.pro")
