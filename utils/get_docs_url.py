def get_docs_url(file_path: str) -> str:
    www_index = file_path.index("www")

    return "https://" + file_path[www_index:]


if __name__ == "__main__":
    print(get_docs_url("documents/drf-docs/www.django-rest-framework.org/api-guide"))
