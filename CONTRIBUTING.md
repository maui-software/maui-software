# Contributors Guide
Hello! If you are reding this, you probably desire to
contribute with maui-software. Thank you so much!

Here you can find some steps to conributing with maui.

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

## Code Style


To ensure maintainability and straightforward understanding of the code,
this project adheres to PEP8 standards. Pull requests will be accepted
only if they pass code style validations by Codacy and Code Rabbit.

To locally check compliance with the standards, pylint and black libraries
are installed in dev group. Use the following commands verify and modify
yout code:

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


## Pull requests and branches

It is not allowed to push new developments right into main branch. 
You should create a new branch, commit and push 
that new branch and just then create a new Pull request in GitHub. 
If the Pull request is accepted, it will be merged into dev branch, 
and will be part of the package in the next release.

Remember that the branch name should indicate the objective of that branch.
