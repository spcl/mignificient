#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "MIG configuration requires root privileges! Will try running with sudo"
	use_sudo=true;
else
	use_sudo=false;
fi

run_cmd()
{
  if [ "$EXEC_MODE" = false ]; then
    "$@"
  else
    sudo "$@"
  fi
}

get_gpu_instance_id()
{
  case $1 in
    "1g") echo "19" ;;
    "1g.10") echo "15" ;;
    "2g") echo "14" ;;
    "3g") echo "9" ;;
    "4g") echo "5" ;;
    "7g") echo "0" ;;
    *) echo "Invalid profile"; exit 1 ;;
  esac
}

create_mig_instances()
{
  local gpu_id=$1
  shift
  local configs=("$@")

  echo "Disable MIG mode"
  run_cmd nvidia-smi -i $gpu_id -mig 0

  echo "Enable MIG mode"
  run_cmd nvidia-smi -i $gpu_id -mig 1

  for config in "${configs[@]}"; do

		echo "Create partition $config"

    gpu_instance_id=$(get_gpu_instance_id "$config")
    
    echo "Create MIG instance $config"
    run_cmd nvidia-smi mig -i $gpu_id -C -cgi $gpu_instance_id
  done
}

# Main script
if [ $# -lt 2 ]; then
  echo "Usage: $0 <gpu-id> <config1> <config2> ..."
  echo "Example: $0 0 1g 3g 4g"
  exit 1
fi

# Call the function to create MIG instances
create_mig_instances "$@"

echo "MIG configuration completed successfully."
