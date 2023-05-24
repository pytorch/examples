# Setup AWS cluster with pcluster
Refer https://www.hpcworkshops.com/04-pcluster-cli.html

## 1. Sign in to an AWS instance

## 2. Install pcluster
```
pip3 install awscli -U --user
pip3 install "aws-parallelcluster" --upgrade --user
```

## 3. Create a cluster config file
```
pcluster configure --config config.yaml
```
See config.yaml.template for an example. Ensure you have a valid EC2 key-pair file


## 4. Create the cluster
```
pcluster create-cluster --cluster-name dist-ml --cluster-configuration config.yaml
```

### 4a. Track progress
```
pcluster list-clusters
```

## 5. Login to cluster headnode
```
pcluster ssh --cluster-name dist-ml -i your-keypair-file
```

## 6. Install dependencies
```
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv /shared/venv/
source /shared/venv/bin/activate
pip install wheel
echo 'source /shared/venv/bin/activate' >> ~/.bashrc
```

## 7. Download training code and install requirements
```
cd /shared
git clone --depth 1 https://github.com/pytorch/examples;
cd /shared/examples
git filter-branch --prune-empty --subdirectory-filter distributed/minGPT-ddp
python3 -m pip install setuptools==59.5.0
pip install -r requirements.txt
```
