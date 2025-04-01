import git
from huggingface_hub import snapshot_download
from pathlib import Path


def get_git_root(path=__file__):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def main():
    # DOWNLOAD TO REASONABLE "DEFAULT" DIRECTORY
    DATA_DIR: str = Path(get_git_root()) / 'data'
    snapshot_download(repo_id='dmp2023/presto-data',
                      repo_type='dataset',
                      local_dir=DATA_DIR,
                      max_workers=8,
                      token="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",
                      force_download=True,
                      resume_download=True,
                      etag_timeout=3600)


if __name__ == '__main__':
    main()
