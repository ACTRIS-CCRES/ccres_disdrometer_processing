# Developper guide

## Realease a new version

On `main`

1. Update the Changelog
2. Bump version

    ```bash
    bump-my-version bump [major|minor|patch]
    ```

3. Push

    ```bash
    git push origin main
    git push --tags
    ```

4. Create a new release on GitHub
