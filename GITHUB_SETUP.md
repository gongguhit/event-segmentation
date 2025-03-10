# Setting Up GitHub Repository

Follow these steps to set up your GitHub repository for this project.

## 1. Initialize Git Repository (First Time)

```bash
# Navigate to your project directory
cd path/to/event_segmentation

# Initialize a git repository
git init

# Add all files to git (except those in .gitignore)
git add .

# Commit these files
git commit -m "Initial commit"
```

## 2. Create a Repository on GitHub

1. Go to [GitHub](https://github.com)
2. Log in to your account
3. Click the "+" icon in the top-right corner and select "New repository"
4. Name your repository (e.g., "event-segmentation")
5. Provide a description (optional)
6. Choose public or private visibility
7. Do NOT initialize with README, .gitignore, or license (since we've already created these)
8. Click "Create repository"

## 3. Link Your Local Repository to GitHub

GitHub will show you commands to connect your existing repository. Use these:

```bash
# Add your GitHub repository as a remote
git remote add origin https://github.com/yourusername/event-segmentation.git

# Push your code to GitHub
git push -u origin main  # or 'master' depending on your default branch name
```

## 4. Updating Your Repository

After making changes:

```bash
# See what files were changed
git status

# Add changes to staging area
git add .

# Commit changes
git commit -m "Your detailed commit message describing changes"

# Push changes to GitHub
git push
```

## 5. Managing Large Files

The MVSEC dataset files are large and shouldn't be committed to GitHub. They are already in the .gitignore file.

If you need to share large files with collaborators, consider using:
- [Git LFS](https://git-lfs.github.com/) for large file storage
- [Academic Torrents](https://academictorrents.com/) for dataset distribution
- Cloud storage services (Google Drive, Dropbox) with links in your README

## 6. Creating Releases

When you have stable versions of your code:

1. Go to your GitHub repository
2. Click "Releases" on the right sidebar
3. Click "Create a new release"
4. Tag a version (e.g., v1.0.0)
5. Add release notes
6. Publish release 