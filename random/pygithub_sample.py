from github import Github

g = Github("myo-frobom", "mypassword")

for repo in g.get_user().get_repos():
    print(repo.name)