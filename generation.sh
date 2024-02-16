'''
 Should Change: 

 1. Change meta_path
 2. item in total (conduct different NLP task and choose different prompts)
 3. args.message_path. (Format: Like in data/beaver_*)
 4. args.result_path (the path to store the generated result.

'''

meta_path=YOUR_META_PATH

total=(
    'summarize Summarize-this-article: front prompt-1'   
)

for((j=0;j<1;j++))do
    model=${models[j]}
    for((i=0;i<1;i++))do

        # get prefix
        cur=${total[i]}
        inner_array=($cur)
        task=${inner_array[0]}
        prompt=${inner_array[1]}
        prompt_type=${inner_array[2]}
        prefix=${inner_array[3]}

        # get device
        device=${devices[i]}

        log_dir=$meta_path/multi_logs/$task
        if [ -d "$log_dir" ]; then
            echo "log dir exist."
        else
            echo 'log dir does not exist.'
            echo $log_dir
            mkdir -p $log_dir
        fi
        # 修改log_path
        log_path=$log_dir/llama_${task}_${model}_${prefix}_total-0.25.log
        echo $log_path

        CUDA_VISIBLE_DEVICES=$device nohup python $meta_path/multi_prompt_result.py \
        --model-path meta-llama/Llama-2-${model}-chat-hf \
        --device 'cuda' \
        --num-gpus 1 \
        --gpus 0 \
        --use-system-prompt True \
        --max-gpu-memory '48Gib' \
        --prompt-type $prompt_type \
        --task $task \
        --prompt $prompt \
        --message-file $meta_path/data/DATA_FILE_NAME \
        --result-file $meta_path/final_results/multi/total/${task}/llama2-${model}-0.25-${prefix}-results.json >$log_path 2>&1 &
    done
done