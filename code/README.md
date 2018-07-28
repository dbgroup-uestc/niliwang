The code is based on paper "[Knowledge Transfer for Out-of-Knowledge-Base Entities: A Graph Neural Network Approach](https://arxiv.org/abs/1706.05674)" ande code [here](https://github.com/takuo-h/GNN-for-OOKB/blob/master/main.py).<br>

###Requirements<br>
pytorch v0.4<br>

###Training<br>
main.py --batch_size 5000 --nn_model A --l_rate 0.004 --sample_size 25 --lstm_activate relu --device 0 --threshold 1200<br>
