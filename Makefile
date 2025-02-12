SHELL = /bin/bash

SYSTEM_NAME := $(shell uname)
SYSTEM_ARCHITECTURE := $(shell uname -m)
MAMBA_INSTALL_SCRIPT := Miniforge3-$(SYSTEM_NAME)-$(SYSTEM_ARCHITECTURE).sh

MAMBA_ENV_NAME := handdgp
PACKAGE_FOLDER := src/handdgp

# HELP: install-mamba: Install Mamba
.PHONY: install-mamba
install-mamba:
	@echo "Installing Mamba..."
	@curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$(MAMBA_INSTALL_SCRIPT)"
	@chmod +x "$(MAMBA_INSTALL_SCRIPT)"
	@./$(MAMBA_INSTALL_SCRIPT)
	@rm "$(MAMBA_INSTALL_SCRIPT)"

# HELP: create-mamba-env: Create a new Mamba environment
.PHONY: mamba-env
mamba-env:
	@mamba env create -f environment.yml -n "$(MAMBA_ENV_NAME)"
	@echo -e " Mamba env created!"
	@echo -e "🎉🎉 Your new $(MAMBA_ENV_NAME) mamba environment is ready to be used 🎉🎉"
