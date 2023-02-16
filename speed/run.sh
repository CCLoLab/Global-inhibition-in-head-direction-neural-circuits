flysim="./flysim07_21_4.out"

python3 gen_conf_pro.py
$flysim -conf network.conf -pro network.pro -nmodel LIF
python3 plot.py

