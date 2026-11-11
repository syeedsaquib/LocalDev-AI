from main import ProjectManager

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild project index")
    parser.add_argument("--project-id", required=True)
    args = parser.parse_args()
    print(ProjectManager.index_project(args.project_id))
