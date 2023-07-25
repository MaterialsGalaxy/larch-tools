# larch-tools
Galaxy tool wrappers for Larch analysis tools for X-ray spectroscopy

## Run/test tools locally
These tools are developed using [Planemo](https://github.com/galaxyproject/planemo) and can be run and tested locally using Planemo without installing Galaxy.

Create a virtualenv and install Planemo:
```bash
$ virtualenv .venv
$ . .venv/bin/activate
$ pip install planemo
```

Clone this repository, then go into the `larch-tools` folder.

To run the tools:
```bash
$ planemo serve
```
To test the tools:
```bash
$ planemo test
```

## Contributing

To get started, see the [Galaxy tool development tutorial](https://training.galaxyproject.org/training-material/topics/dev/tutorials/tool-integration/slides.html).

When writing tools, follow the [Intergalactic Utilities Commission tool standards](https://galaxy-iuc-standards.readthedocs.io/en/latest/index.html).

Use Planemo to lint and test tools; see the [Planemo documentation](https://planemo.readthedocs.io/en/latest/index.html) for more information.