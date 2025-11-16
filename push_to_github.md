# Push to GitHub - Instructions

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `spam-email-detection` (or your preferred name)
3. Description: "Complete spam email detection system using NLP and Machine Learning"
4. Choose Public or Private
5. **DO NOT** check "Initialize with README"
6. Click "Create repository"

## Step 2: Push to GitHub

After creating the repository, run these commands (replace `YOUR_USERNAME` with your GitHub username):

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/spam-email-detection.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Alternative: Using SSH

If you have SSH keys set up:

```bash
git remote add origin git@github.com:YOUR_USERNAME/spam-email-detection.git
git branch -M main
git push -u origin main
```

## Quick One-Liner (after creating repo on GitHub)

Replace `YOUR_USERNAME` and `REPO_NAME`:

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git && git branch -M main && git push -u origin main
```

