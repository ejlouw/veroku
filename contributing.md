### Contributing
<div style="text-align: justify">
If you would like to contribute to veroku, the TODO's in the code are a good place to start. There are a some very simple
ones, but also a few more complex ones. Another area where contributions will be valuable is in the completion and
refinement of the experimental modules and writing tests for these modules. Another potential area for contribution
would be the items on the roadmap, although it would be best if the experimental modules are first rounded off.

This project uses gitflow for branch management. Below is a summary of how to make a contribution to this project using
the gitflow process:
* Fork this repository to your own GitHub account.
* Clone the forked repository to make it available locally.
* Switch to the dev branch (`git checkout dev`) and branch of of this branch to create your new feature branch 
(`git checkout -b feature/<your-feature-name>`).
* Make changes and add features, commit these changes when done and push the changes to your GitHub repo.
* Create a pull request to request that your changes be added to veroku.

In general, please remember to ensure that the following guidelines are followed when contributing:
</div>

* Run `pylint ./veroku` from the project root to ensure that there are no code style issues (this will cause the actions
 pipeline to fail)
* The use of pylint disable statements should be reserved only for special cases and are not generally acceptable.
* Add tests for any contributions ( this will also prevent the build from failing on the code coverage check).
* Run all tests before pushing.	
