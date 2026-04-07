# Linking data from projects

It is good practice to keep the `data` directory here fixed, and use in your code paths
relative to this folder, so that they never change.

You will need to create soft links to be able to connect whatever folder inside data.
For this you can use the `ln` command, like so:

> Note: Launch this command from the **project root**.

```shell
$ln -s /nfs/projects/whatever-folder/ data/
```

This will create a subdirectory `whatever-folder` *inside* `data/`

> **Make sure to keep the gitignore file, or in general to ignore the contents of this folder!**