#/bin/bash

config_file=$1
config_path=$(dirname $config_file)

if [ -f "$config_file" ]; 
then
	source $config_file
else
	echo -e "ERROR: Configuration file does NOT exist!\nExiting..."
	exit 1
fi

guppy_command_array=(guppy_basecaller -i ${sequencing_raw_data} -s ${basecaller_output_data} --flowcell ${basecaller_flowcell} --kit ${basecaller_kit})
minimap_command_array=(minimap2 -x map-ont -c -N 1 ${minimap_index_data} ${minimap_input_data} -o ${minimap_output_data})
selection_evaluator_command_array=(python3 ${selection_evaluator_path}/selection_evaluator.py --sequencing-output ${selection_simulation_output} --fast5-reads ${sequencing_raw_data} --minimap-output ${minimap_output_data})

echo -e "Executing Guppy command...\n"
"${guppy_command_array[@]}"

echo -e "\nExecuting Minimap command...\n"
"${minimap_command_array[@]}"

echo -e "\nEvaluating selective sequencing...\n"
"${selection_evaluator_command_array[@]}"
