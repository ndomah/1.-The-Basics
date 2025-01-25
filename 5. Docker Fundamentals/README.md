# Docker Fundamentals

## Docker Concepts

### Docker vs. Virtual Machines

![docker vs vms](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/img/fig1%20-%20vm%20vs%20docker.jpg)

- **Docker** is a platform that enables developers to package applications and their dependencies into lightweight, portable containers that run consistently across different environments

||**Virtual Machines**|**Docker**|
|---:|---|---|
|**Architecture**|Each VM includes a full guest OS and runs on a hypervisor|Containers share the host OS and run on the Docker Engine, making them more efficient|
|**Resource Usage**|Require more memory and CPU due to multiple OS instances|Lightweight, sharing the OS kernel and using fewer resources|
|**Speed & Portability**|Slow startup and less portable|Fast, portable across environments, and ideal for CI/CD workflows|
|**Isolation**|Strong isolation with separate OS instances|Process-level isolation, sufficient for most applications	|

**Why use Docker?**
-	Faster deployment and scaling
-	Lower resource consumption
-	Simplified dependency management
-	Works across various environments consistently

### Docker Terminology

![docker fundamentals](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/img/fig2%20-%20docker%20fundamentals.png)

**Code**
-	The application code and dependencies that you want to package and deploy.
-	Using a `Dockerfile`, code is built into an image.
**Image**
-	A Docker image is a lightweight, standalone, and executable package that includes everything needed to run a piece of software, such as code, runtime, libraries, and dependencies.
-	Built from the application code and stored locally or in a registry.
-	You can create and run containers from an image.
**Registry**
-	A storage location where Docker images are pushed (uploaded) and pulled (downloaded).
-	Examples: Docker Hub, AWS ECR, GitHub Container Registry.
**Tag**
-	A label assigned to a Docker image to differentiate between versions.
-	Example: `myapp:v1.0`, where `v1.0` is the tag.
**Container**
-	A running instance of a Docker image with a unique name and ID.
-	Containers can have configurations such as environment variables, volumes, ports, and networks.
**Environmental Variables**
-	Configuration values that can be passed to a running container to customize its behavior without modifying the image.
**Volumes**
-	Persistent storage that allows data to be shared between the host system and the container or between multiple containers.
**Ports**
-	Containers expose ports to allow external systems to communicate with the services running inside the container.
-	Example: Mapping port `8080` inside the container to `80` on the host.
**Networks**
-	Defines how containers communicate with each other and the outside world.

**Docker Workflow Overview**:
1.	**Build**: Code is used to create an image.
2.	**Tag**: The image is labeled with a version.
3.	**Push/Pull**: Images can be pushed to and pulled from a registry.
4.	**Run**: The image is used to create a container.
5.	**Configure**: The container is assigned environment variables, storage, ports, and network settings.

## Hands-On

### Pulling Images & Running Containers in CLI

-	We will use the [Portainer Community Edition](https://hub.docker.com/r/portainer/portainer-ce): 
-	Open Powershell and run the Docker Pull Command: `docker pull portainer/portainer-ce`
-	From the [Portainer CE Server Installation](https://docs.portainer.io/start/install-ce/server/docker/wsl): 
  -	Using Docker standalone we will install Portainer CE on WSL/Docker Desktop
  -	First, create the volume that Portainer Server will use to store its database:
    -	`docker volume create portainer_data`
  -	Then, download and install the Portainer Server container:
    -	`docker run -d -p 8000:8000 -p 9443:9443 --name portainer --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce:2.21.5`
    -	We will make some changes to this command
      -	change `2.21.5` to `latest`
      -	delete `-d`,  we don’t want the container to run detached from the command line
      -	delete `--restart=always`, we don’t want to restart the containers every time we restart windows
   
![portainer cli](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/img/fig3%20-%20portainer%20cli.png)

  - Access Portainer UI on [https://localhost:9443](https://localhost:9443)

### CLI Cheate Sheet

-	Official Docker [CLI Cheat Sheet](https://dockerlabs.collabnix.com/docker/cheatsheet/) 
-	Also refer to [docker_cheatsheet.pdf](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/files/docker_cheatsheet.pdf)

### Docker Compose Explained

-	Refer to [docker-compose.yml](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/files/docker-compose.yml)

```dockerfile
version: '3'

services:
  
  portainer:
    image: portainer/portainer-ce:2.11.1
    container_name: portainer
    restart: on-failure
    ports:
      - 9443:9443
      - 9000:9000
      - 8000:8000
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - portainer_data:/data
volumes:
  portainer_data:
```

-	It contains similar tasks to what we did before in the CLI
-	Back to hands-on:
  -	Run `docker stop portainer`to stop portainer
  -	Verify with `docker ps`, should be empty
  -	Change directory to location of yaml file and run `docker compose up`
  -	This won’t work because we only stopped the container, we must delete it too: `docker rm portainer`
  -	Run `docker compose up` again

![yaml cli](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/img/fig4%20-%20yaml%20cli.png)

- Access Portainer UI on [https://localhost:9443](https://localhost:9443) and add password

![portainer ui](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/img/fig5%20-%20portainer%20ui.png)

- Hit `ctrl + c` on command line to quit

### Build and Run a Simple Hello World Image

- Refer to [hello-world.py]()

```python

```

- Refer to [Dockerfile]()

```docker

```

- Go back to command line and change to directory where python folder is stored
-	Run `docker build -t hello-world .`
-	Run `docker run hello-world`
-	Run `docker run hello-world test` to test it with an argument

![hello world](https://github.com/ndomah/1.-The-Basics/blob/main/5.%20Docker%20Fundamentals/img/fig5%20-%20hello%20world.png)
