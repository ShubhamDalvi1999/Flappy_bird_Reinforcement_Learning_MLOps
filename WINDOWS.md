# Running Flappy Bird RL on Windows

This document provides specific instructions for running the Flappy Bird Reinforcement Learning project on Windows environments.

## Prerequisites

1. **Docker Desktop for Windows**
   - Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
   - Make sure to enable the WSL 2 backend during installation
   - After installation, verify Docker is running with the command: `docker --version`

2. **Windows Terminal** (recommended)
   - For a better command-line experience, consider installing [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)

3. **Git for Windows** (optional)
   - If you want to use Git with the project, install [Git for Windows](https://gitforwindows.org/)

## Setting Up the Project

1. Clone or download the project to your local machine

2. Open Command Prompt or PowerShell in the project directory

3. If you want to use Weights & Biases for tracking, set your API key:
   ```powershell
   set WANDB_API_KEY=your_api_key_here
   ```

## Running the Application

We've provided a Windows batch script (`run.bat`) that makes it easy to manage the application:

### Starting the Services

```cmd
run.bat start
```

This will start all the Docker containers. The following services will be available:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000/api
- MLflow UI: http://localhost:5001
- Spark Master UI: http://localhost:8080

### Other Commands

- Stop all services: `run.bat stop`
- Restart all services: `run.bat restart`
- View logs: `run.bat logs`
- Check service status: `run.bat status`
- Run tests: `run.bat test`
- Clean up (remove containers and volumes): `run.bat clean`
- Show help: `run.bat help`

## Troubleshooting

### Docker Volume Issues

If you encounter issues with Docker volumes on Windows:

1. Open Docker Desktop
2. Go to Settings > Resources > File Sharing
3. Make sure the drive containing your project is shared with Docker

### Port Conflicts

If you see errors about ports being in use:

1. Check if any other applications are using ports 3000, 5000, 5001, or 8080
2. Stop those applications or modify the ports in `docker-compose.yml`

### WSL 2 Issues

If Docker is having trouble with WSL 2:

1. Make sure WSL 2 is properly installed: `wsl --status`
2. Update WSL if needed: `wsl --update`
3. Set WSL 2 as the default: `wsl --set-default-version 2`

### Performance Considerations

To improve performance on Windows:

1. In Docker Desktop, go to Settings > Resources
2. Adjust the memory allocation (at least 4GB recommended for this project)
3. Adjust the number of CPUs (at least 2 recommended)

## Windows-Specific Notes

### Line Endings

Windows and Unix-based systems use different line endings. If you edit files in Windows and then run them in Docker (Linux), you might encounter issues. To avoid this:

1. Configure Git to handle line endings correctly:
   ```bash
   git config --global core.autocrlf input
   ```

2. Use a code editor that supports Unix-style line endings (LF), such as VS Code

### Path Separators

Windows uses backslashes (`\`) for file paths, while the Docker containers (Linux) use forward slashes (`/`). The project is configured to handle this, but be aware when manually modifying paths. 