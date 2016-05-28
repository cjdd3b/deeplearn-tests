'''
fabfile.py

Many of these commands require a boto configuration file to be set up with your
credentials and saved to ~/.boto. The contents should look like this:

[Credentials]
aws_access_key_id = <your_access_key_here>
aws_secret_access_key = <your_secret_key_here>
'''
import time, socket, os
import boto.ec2
from fabric import api
from fabric.state import env
from fabric.colors import green as _green, red as _red, yellow as _yellow

########## SETTINGS ##########

# Globals, set via environment variables

# Required
GPU_INSTANCE_KEY = os.environ.get('GPU_INSTANCE_KEY') # Minus the .pem

# Optional
GPU_INSTANCE_REGION = os.environ.get('GPU_INSTANCE_REGION', 'us-west-1') # N. California
GPU_INSTANCE_AMI_ID = os.environ.get('GPU_INSTANCE_AMI_ID', 'ami-91b077d5') # Community AMI with CUDA, Theano, etc.
GPU_INSTANCE_TYPE = os.environ.get('GPU_INSTANCE_TYPE', 'g2.2xlarge')
GPU_INSTANCE_NAME = os.environ.get('GPU_INSTANCE_NAME', 'deeplearn-gpu') 

# Env settings
env.project_repo = 'git@github.com:newsdev/deeplearn-test.git'
env.user = "ubuntu"
env.forward_agent = True

########## HELPERS ##########

def _get_gpu_host():
    conn = boto.ec2.connect_to_region(GPU_INSTANCE_REGION)
    reservations = conn.get_all_instances(filters={"tag:Name" : GPU_INSTANCE_NAME})
    instances = [i for r in reservations for i in r.instances if i.state == 'running']
    if len(instances) > 1:
        print(_red("Multiple GPU hosts detected. This script only supports one."))
        exit()
    return instances[0] if len(instances) == 1 else []

def _set_env():
    env.instance = _get_gpu_host()
    env.host = env.instance.public_dns_name if env.instance else ''
    return
    
def _launch_gpu():
    '''
    Boots up a new GPU-based instance on EC2.
    '''
    print(_green("Started..."))
    print(_green("Creating EC2 instance..."))

    try:
        # Create new instance
        conn = boto.ec2.connect_to_region(GPU_INSTANCE_REGION)
        reservation = conn.run_instances(
            GPU_INSTANCE_AMI_ID,
            key_name=GPU_INSTANCE_KEY,
            instance_type=GPU_INSTANCE_TYPE)

        # Assumes we're only using one instance
        instance = reservation.instances[0]
        
        # Wait for instance to boot up
        status = instance.update()
        while status == 'pending':
            print(_yellow("Booting instance ..."))
            time.sleep(10)
            status = instance.update()

        # Once instances are alive, do tagging and other post-activation work
        if status == 'running':
            print(_green("Instance booted! Tagging ..."))
            instance.add_tag('Name', GPU_INSTANCE_NAME)

        # Wait until instance is accessible via SSH
        sshable = False
        while sshable == False:
            print(_yellow("Waiting for SSH connection (this might take a minute) ..."))
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.connect((instance.public_dns_name, 22))
                sshable = True
                print(_green("SSH is now accessible!"))
            except socket.error as e:
                pass
            s.close()

        # Wrapup
        print(_green("Done!"))
        print(_green("ID: %s" % instance.id))
        print(_green("Public DNS: %s" % instance.public_dns_name))
    except:
        print(_red('Error creating instance.'))
        raise
    return

def _bootstrap_gpu():
    _set_env()
    print(_green("Bootstrapping (this takes a while) ..."))
    with api.settings(warn_only=True, host_string=env.host):
        api.put(local_path='./dotfiles/.bashrc', remote_path='/home/ubuntu/.bashrc')
        api.run('git clone %s' % env.project_repo)
        api.run('sudo apt-get update')
        api.run('sudo apt-get install python-dev python-setuptools pkg-config liblapack-dev')
        api.run('sudo easy_install pip')
        api.run('sudo -H pip install -r ~/deeplearn-test/requirements.txt')
    print(_green("Done!"))

########## TASKS ##########

@api.task
def gpu_up():
    _set_env()
    if env.host:
        print(_green("GPU instance already running at: " + env.host))
    else:
        _launch_gpu()
        _set_env()
        _bootstrap_gpu()
        print(_green("GPU instance now running at: " + env.host))

@api.task
def gpu_go():
    _set_env()
    if env.instance:
        api.local('ssh %s@%s' % (env.user, env.host))
    else:
        print(_red("No GPU instance running. Try fab gpu_up first."))

@api.task
def gpu_down():
    conn = boto.ec2.connect_to_region(GPU_INSTANCE_REGION)
    _set_env()
    if env.instance:
        conn.terminate_instances(instance_ids=[env.instance.id])
        print(_green("GPU instances terminated."))