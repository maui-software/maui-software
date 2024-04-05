# Contributors Guide
Hello! If you are reading this, you probably desire to
contribute to maui-software. Thank you so much!

Here you can find some steps to contributing with maui.


## Development installation

1. Clone repository.

	```bash
	git clone https://github.com/maui-software/maui-software.git
	cd ~/maui-software
	```

2. Install Poetry

	Poetry is the package management tool used for this project. 
	Refer to the 
	[official website](https://python-poetry.org/docs/#installing-with-pipx) 
	for instructions about how to install Poetry.
	A simple way is the following:

	```bash
	pipx install poetry
	```

3. Create a virtual environment within Poetry

	```bash
	poetry env use python3
	```

4. Install package dependencies

	```bash
	poetry install
	poetry install --all-extras
	```

5. Install git lfs
	Due to the presence of large files within the samples module, we utilize
	Git LFS to manage them effectively. To install Git LFS, please use the
	following commands:


	```bash
	sudo apt update
	sudo apt install git-lfs
	```


## Managing dependencies

If you have to add a new dependency in your development process,
 this dependency should be added using Poetry as follows:

1. Add the dependency to pyproject.toml

	```bash
	poetry add dependency_name
	```

	* If you want to add a dependency just for test or dev environment, use one of the following commands:

	```bash
	poetry add dependency_name --group test
	```

	```bash
	poetry add dependency_name --group dev
	```

	* If you need to remove a dependency, use the following command:

	```bash
	poetry remove dependency_name
	```

	If you need a specific version of some dependency, refer to the [official website](https://python-poetry.org/docs/managing-dependencies/).

2. Update poetry.lock

	```bash
	poetry lock
	```

3. Install dependencies

	```bash
	poetry install --sync
	```

## Making changes

Once your changes are done, you should check the __init__.py
file of the module that you are working on. 
All the methods of the module should be listed in that file. 
If everything is ok, run the following command to install the changes:

```bash
poetry install
```

It is important that all changes are properly tested and works correctly.


## Code Style

To ensure maintainability and straightforward understanding of the code,
this project adheres to PEP8 standards. Pull requests will be accepted
only if they pass code style validations by Codacy and Code Rabbit.

To locally check compliance with the standards, pylint and black libraries
are installed in dev group. Use the following commands to verify and modify
your code:

1. Verify standards:
	
	```bash
	pylint your_file.py
	```
2. Automatic changes:

	```bash
	black your_file.py
	```

This is an iterative process. Periodically check that the developed code
meets PEP8 standards.

## Update documentation

After ensuring that the code works without errors, one should verify if 
the documentation is correctly updated. To do so, locally, 
first navigate to the root directory of the project and run the 
following commands:

```bash
cd docs
make clean-generated-rst clean html
```

This will update the documentation and store it locally under 
`/maui-software/docs/build/html`. 
Verify if the documentation is correct and easy to understand!

## Pull requests and branches

It is not allowed to push new developments right into main branch. 
You should create a new branch, commit and push 
that new branch and just then create a new Pull request in GitHub. 
If the Pull request is accepted, it will be merged into dev branch, 
and will be part of the package in the next release.

Remember that the branch name should indicate the objective of that branch.

## New Releases

Once enough changes has been made and it is time to create a new release,
the following steps should be performed:

1. Commit the new version to the development branch

	The changes should be done in two files:
	- pyproject.toml: change the version tag to the new version
	- docs/source/conf.py: change the release and version to the new version

2. Merge development branch into main branch
3. Create a new tag in github in main branch with the new version
4. Create a new release in github from the created tag

Once this steps were performed, the docs will be automaticaly updated and
the new release will be uploaded to PyPi.

IMPORTANT: Only project admins can create new releases.
