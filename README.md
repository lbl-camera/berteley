# BERTeley
Ushizima working with Mudit Mangal and Eric Chagnon in 2022

## Setting up the environment
If working on a local machine use the following command conda create --name <env name> --file <env.txt>.

If working in NERSC and you want to create a kernel to run the notebook in JupyterLab do the following
1. module load python
2. conda create -n <env name> python=3.7 ipykernel --file <path to env.txt>
3. source activate nlp_env
4. python -m ipykernel install --user --name <env_name> --display-name <display name of env>
