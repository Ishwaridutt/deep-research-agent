# Report rendering dependencies

These pinned browser builds are served locally; report rendering makes no CDN requests.

| Library | Version | Source | License |
| --- | --- | --- | --- |
| Marked | 18.1.1 | https://registry.npmjs.org/marked/-/marked-18.1.1.tgz | MIT; `MARKED-LICENSE` |
| DOMPurify | 3.4.17 | https://registry.npmjs.org/dompurify/-/dompurify-3.4.17.tgz | Apache-2.0 OR MPL-2.0; accompanying license files |

Package tarballs were checked against the npm registry's SHA-512 integrity values before extraction:

- Marked: `sha512-zEGgrtUG8crk+TrMacJSVABU4DYDDkS8uqgU3ohrn/2fW9C2AEfUrO3m6IhpIpQpl7YcVpiXXC9cSMWDANT3dg==`
- DOMPurify: `sha512-B/pSlYVHQ58kqISD/yDjuJ1yfzI34m1gF4lAod2bwSRhiN4K5Cl3H8Af7hqB2PmTL911tdEudnnXn5FnhRoz0w==`

Recheck upstream security advisories when updating. Do not insert unsanitized Marked output into the DOM.
