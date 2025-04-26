export PYTHONPATH=./

#./scripts/run_fan_wandb.sh "DLinear" "No FAN"  "ETTh1 " "96 168 336 720"  "cuda:0" 96  "{freq_topk:4}"
#!/bin/bash

declare -A dataset_to_window_map

#"Weather" "ETTh1" "ExchangeRate" "Electricity" "Traffic"
#"96" "168" "336" "720"

models=("DLinear" "SCINet")
norms=("FAN")  

datasets=("Weather" "ExchangeRate")
pred_lens=("96" "168" "336" "720")
device="cuda:0"
windows=96
norm_config="{freq_topk:2}"
for model in "${models[@]}"
    do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py --dataset_type="$dataset" --norm_type="$norm" --norm_config=$norm_config --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100  runs --seeds='[2024]'
            done
        done
    done
done

datasets=("ETTh1")
pred_lens=("96" "168" "336" "720")
windows=96
norm_config="{freq_topk:4}"
for model in "${models[@]}"
    do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py --dataset_type="$dataset" --norm_type="$norm" --norm_config=$norm_config --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100  runs --seeds='[2024]'
            done
        done
    done
done


datasets=("Electricity")
pred_lens=("96" "168" "336" "720")
windows=96
norm_config="{freq_topk:3}"
for model in "${models[@]}"
    do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py --dataset_type="$dataset" --norm_type="$norm" --norm_config=$norm_config --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100  runs --seeds='[2024]'
            done
        done
    done
done


datasets=("Traffic")
pred_lens=("96" "168" "336" "720")
windows=96
norm_config="{freq_topk:30}"
for model in "${models[@]}"
    do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py --dataset_type="$dataset" --norm_type="$norm" --norm_config=$norm_config --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100  runs --seeds='[2024]'
            done
        done
    done
done

echo "All runs completed."

