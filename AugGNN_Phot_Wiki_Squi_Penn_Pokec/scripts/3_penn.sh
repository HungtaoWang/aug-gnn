python main.py --model ebdgnn --dataset penn --lr 1e-2  --weight_decay 1e-5  --dropout 0.1 --num_layers 2 --augment_sp 1 --sp_rate 0.9  --se_type 'rwr' --se_dim 1024 --weigh 0.2  --fi_type 'ori' --si_type 'se' --gnn_type "gcn" --sw 1.0   --device cuda:0 --times 10 --device cuda:0 --early_stopping 500 
python main.py --model ebdgnn --dataset penn --lr 5e-3  --weight_decay 1e-5  --dropout 0.1 --num_layers 2 --augment_sp 1 --sp_rate 0.5  --se_type 'rwr' --se_dim 128 --weigh 0.2  --fi_type 'ori' --si_type 'se' --gnn_type "gat" --sw 1.0   --device cuda:0 --times 10 --device cuda:0 --early_stopping 500 
python main.py --model ebdgnn --dataset penn --lr 5e-2  --weight_decay 1e-5  --dropout 0.1 --num_layers 2 --augment_sp 1 --sp_rate 0.5  --se_type 'rwr' --se_dim 128 --weigh 0.2 --fi_type 'ori' --si_type 'se' --gnn_type "sgc" --sw 0.5   --device cuda:0 --times 10 --device cuda:0 --early_stopping 500
python main.py --model ebdgnn --dataset penn --lr 5e-2  --weight_decay 1e-5  --dropout 0.1 --num_layers 2 --augment_sp 1 --sp_rate 0.5  --se_type 'rwr' --se_dim 128 --weigh 0.2  --fi_type 'ori' --si_type 'se' --gnn_type "appnp" --sw 0.5 --device cuda:0 --times 10 --device cuda:0 --early_stopping 500 
python main.py --model ebdgnn --dataset penn --lr 1e-2  --weight_decay 1e-5  --dropout 0.1 --num_layers 2 --augment_sp 1 --sp_rate 0.9  --se_type 'rwr' --se_dim 128 --weigh 0.2  --fi_type 'ori' --si_type 'se' --gnn_type "gcn2" --sw 0.5   --device cuda:0 --times 10 --device cuda:0 --early_stopping 500 




