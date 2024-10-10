# Mi Fitness Analyser

Mi Fitness analysis of exported data.

## Dependencies

* [Docker](https://docs.docker.com) (optional): Used to develop insider Docker development container.

## Project setup

First step is to clone the repository.

```shell
git clone 
```

After having cloned the repo, there are 2 ways to manage the development environment:

* Open de project in a development container with VSCode.
* Create a virtual environment and install Python packages:

```shell
make venv
```

To configure the coverage threshold, go to the ``.coveragerc`` file.

## Poetry

To include a new package to the project it should be added to ``pyproject.toml`` under the correct group:

* Packages needed to run the applications should be under the ``tool.poetry.dependencies`` section.
* Packages used as development tools such as ``pytest``, ``ruff`` or ``black`` belong to the ``tool.poetry.group.dev.dependencies`` section.

To add a package you can use ``poetry add``. You can indicate the group to add the dependency to with the option ``--group=GROUP``.

To remove a package use ``poetry remove``.

### poetry.lock

The ``poetry.lock`` file contains a snapshot of the resolved dependencies from ``pyproject.toml``.

To manually force the update of `poetry.lock` file, run ``poetry lock``. The ``--no-update`` flag can be used to avoid updating those dependencies which do not need to.

### Jupyter

You can start a local notebook or jupyter lab server with:

```shell
make jupyter
```

## Testing

To verify correct installation, execute the Mi Fitness Analyser tests by running the following command in your Python environment:

```shell
make tests-basic
```

## Generate documentation

Run the following command to generate project documentation:

```shell
make docs
```

## Contributors

- Contributor name ([contributor email](mailto:contributor email))
