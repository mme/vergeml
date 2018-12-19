Projects
============

Your starting point in VergeML is a project. In general, it is a directory containing all relevant files for your ML project. A project could, for example, be an image classifier to classify plants or a sentiment analysis in product reviews. Whatever it is, your project will be based on data, a model and training sets.

To create a new project, without specifying an underlying model, type: 

    ml new project_name

And if you already know what ML model you want to use, include the model name in your command like this: 

    ml new project_name --model=model-name

To see what models are currently supported in VergeML, visit [Models](/Models/Models.md)

To summarize, a project will handle these tasks and topics for you:
* Create and manage required direcotries for your samples, training, caching, etc.
* Create and manage your configuration file(s) to make your journey in VergeML easier.
* Store and manage your trained AIs.

Next read
============

If you want to continue reading, we suggest this as your next topic: [Configuration File](/Training/Configuration.md)

or, alternatively, jump to the [Table of Content](/TOC.md)