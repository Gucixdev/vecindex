/* vserv_tls_auto.c - TLS with nothing to configure, included by vserv.c (TLS builds only).
 *
 * Without -c/-k, a server with a TCP port makes its own key and certificate the first time it starts: an EC P-256
 * key from the system's random source, a self-signed certificate around it (CN vserv, valid until 9999, its own CA,
 * the names localhost, 127.0.0.1, ::1 and the host name), written as
 * vserv.key (mode 0600) and vserv.crt next to the store (or into the tenant directory) and used from then on. Its
 * fingerprint - SHA-256 of the certificate's SubjectPublicKeyInfo, as curl's --pinnedpubkey writes it - is printed at
 * every start: `sha256//BASE64`. Clients pin it instead of trusting a CA: curl -k --pinnedpubkey sha256//..., and a
 * replica (vserv -F) remembers it on its first connection, the way ssh remembers a host key, and refuses a server that
 * shows another one. */

#ifdef VSERV_TLS

/* ---- DER, the few shapes a certificate needs ---- */

/* tag, length, content: the bytes at o (when not NULL); returns how many */
static size_t vs_der_(unsigned char *o, u8 tag, const unsigned char *c, size_t n)
{
    size_t h = n < 128 ? 2 : (n < 256 ? 3 : 4);
    if (o) {
        if (n) memmove(o + h, c, n);                             /* first: c may be o itself (wrapping in place) */
        o[0] = tag;
        if (n < 128) o[1] = (unsigned char)n;
        else if (n < 256) { o[1] = 0x81; o[2] = (unsigned char)n; }
        else { o[1] = 0x82; o[2] = (unsigned char)(n >> 8); o[3] = (unsigned char)(n & 0xff); }
    }
    return h + n;
}

/* the element at p[0, n): its tag in *tag, the start of its content in *c and the content's length in *cn; returns
 * the element's whole length, 0 when it does not fit in n or is not DER this file reads */
static size_t vs_der_get_(const unsigned char *p, size_t n, u8 *tag, const unsigned char **c, size_t *cn)
{
    size_t h = 2, len, i;
    if (n < 2) return 0;
    *tag = p[0];
    if ((p[0] & 0x1f) == 0x1f) return 0;                         /* tag numbers past 30: no certificate uses them here */
    if (p[1] < 0x80) len = p[1];
    else {
        size_t k = p[1] & 0x7f;
        if (k < 1 || k > 3 || n < 2 + k) return 0;
        for (len = 0, i = 0; i < k; ++i) len = (len << 8) | p[2 + i];
        h += k;
    }
    if (len > n - h) return 0;
    *c = p + h; *cn = len;
    return h + len;
}

/* the SubjectPublicKeyInfo of the certificate der[0, n), as it is in it: 0 with *spki and *len, or -1 */
static i32 vs_spki_(const unsigned char *der, size_t n, const unsigned char **spki, size_t *len)
{
    const unsigned char *c, *t;
    size_t cn, tn, k, i;
    u8 tag;
    if (!vs_der_get_(der, n, &tag, &c, &cn) || tag != 0x30) return -1;        /* Certificate */
    if (!vs_der_get_(c, cn, &tag, &t, &tn) || tag != 0x30) return -1;         /* TBSCertificate */
    for (i = 0; i < 7; ++i) {                    /* [0] version (optional), serial, signature, issuer, validity, subject, key */
        const unsigned char *e;
        size_t en;
        if (!(k = vs_der_get_(t, tn, &tag, &e, &en))) return -1;
        if (i == 0 && tag != 0xa0) ++i;          /* a v1 certificate: no version */
        if (i == 6) { if (tag != 0x30) return -1; *spki = t; *len = k; return 0; }
        t += k; tn -= k;
    }
    return -1;
}

static void vs_b64_(char *o, const unsigned char *p, size_t n)
{
    static const char a[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t i;
    for (i = 0; i + 2 < n; i += 3) {
        *o++ = a[p[i] >> 2]; *o++ = a[((p[i] & 3) << 4) | (p[i + 1] >> 4)];
        *o++ = a[((p[i + 1] & 15) << 2) | (p[i + 2] >> 6)]; *o++ = a[p[i + 2] & 63];
    }
    if (n - i == 1) { *o++ = a[p[i] >> 2]; *o++ = a[(p[i] & 3) << 4]; *o++ = '='; *o++ = '='; }
    else if (n - i == 2) { *o++ = a[p[i] >> 2]; *o++ = a[((p[i] & 3) << 4) | (p[i + 1] >> 4)]; *o++ = a[(p[i + 1] & 15) << 2]; *o++ = '='; }
    *o = 0;
}

/* "sha256//BASE64" of the certificate's public key into pin[61]: 0, or -1 when its DER cannot be read */
static i32 vs_pin_of_(const unsigned char *der, size_t n, char *pin)
{
    const unsigned char *k;
    size_t kn;
    unsigned char h[32];
    br_sha256_context sc;
    if (vs_spki_(der, n, &k, &kn) != 0) return -1;
    br_sha256_init(&sc); br_sha256_update(&sc, k, kn); br_sha256_out(&sc, h);
    memcpy(pin, "sha256//", 8);
    vs_b64_(pin + 8, h, 32);
    return 0;
}

static i32 vs_write_new_(const char *path, const char *p, size_t n, int mode)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
    i32 r = 0;
    if (fd < 0) return -1;
    if (write(fd, p, n) != (ssize_t)n || fsync(fd) != 0) r = -1;
    if (close(fd) != 0) r = -1;
    return r;
}

/* PEM of der[0, n) under banner, malloc'd */
static char *vs_pem_of_(const unsigned char *der, size_t n, const char *banner)
{
    size_t m = br_pem_encode((void *)0, der, n, banner, BR_PEM_LINE64);
    char *o = (char *)malloc(m + 1);
    if (o) br_pem_encode(o, der, n, banner, BR_PEM_LINE64);
    return o;
}

/* one extension: OID, critical or not, the value's DER */
static size_t vs_ext_(unsigned char *o, const unsigned char *oid, size_t on, i32 crit, const unsigned char *v, size_t vn)
{
    unsigned char e[400];
    size_t n = vs_der_(e, 0x06, oid, on);
    if (crit) n += vs_der_(e + n, 0x01, (const unsigned char *)"\xff", 1);
    n += vs_der_(e + n, 0x04, v, vn);
    return vs_der_(o, 0x30, e, n);
}

/* The extensions strict clients (Python's ssl, OpenSSL's X509_STRICT) want of a certificate that is its own CA: basic
 * constraints CA, key usage (signatures, signing certificates), both key identifiers, and the names it answers to -
 * localhost, 127.0.0.1, ::1 and this machine's host name - so a client may also take vserv.crt as its CA file. */
static size_t vs_tls_exts_(unsigned char *o, const unsigned char *pt, size_t pn)
{
    static const unsigned char o_bc[] = { 0x55, 0x1d, 0x13 }, o_ku[] = { 0x55, 0x1d, 0x0f }, o_ski[] = { 0x55, 0x1d, 0x0e },
                               o_aki[] = { 0x55, 0x1d, 0x23 }, o_san[] = { 0x55, 0x1d, 0x11 };
    static const unsigned char bc[] = { 0x30, 0x03, 0x01, 0x01, 0xff };              /* cA TRUE */
    static const unsigned char ku[] = { 0x03, 0x02, 0x02, 0x84 };                    /* digitalSignature, keyCertSign */
    static const unsigned char ip4[] = { 127, 0, 0, 1 }, ip6[] = { 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1 };
    unsigned char id[20], v[400], w[300], all[800];
    char host[256];
    size_t n = 0, vn, wn, i, hl;
    br_sha1_context h1;
    br_sha1_init(&h1); br_sha1_update(&h1, pt, pn); br_sha1_out(&h1, id);         /* RFC 5280 4.2.1.2, method 1 */
    n += vs_ext_(all + n, o_bc, sizeof(o_bc), 1, bc, sizeof(bc));
    n += vs_ext_(all + n, o_ku, sizeof(o_ku), 1, ku, sizeof(ku));
    vn = vs_der_(v, 0x04, id, 20);
    n += vs_ext_(all + n, o_ski, sizeof(o_ski), 0, v, vn);
    wn = vs_der_(w, 0x80, id, 20); vn = vs_der_(v, 0x30, w, wn);
    n += vs_ext_(all + n, o_aki, sizeof(o_aki), 0, v, vn);
    wn = vs_der_(w, 0x82, (const unsigned char *)"localhost", 9);
    wn += vs_der_(w + wn, 0x87, ip4, 4);
    wn += vs_der_(w + wn, 0x87, ip6, 16);
    if (gethostname(host, sizeof(host)) == 0) {
        host[sizeof(host) - 1] = 0;
        hl = strlen(host);
        for (i = 0; i < hl; ++i) {
            char ch = host[i];
            if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '.')) break;
        }
        if (hl && i == hl && hl < 200 && strcmp(host, "localhost")) wn += vs_der_(w + wn, 0x82, (const unsigned char *)host, hl);
    }
    vn = vs_der_(v, 0x30, w, wn);
    n += vs_ext_(all + n, o_san, sizeof(o_san), 0, v, vn);
    n = vs_der_(v, 0x30, all, n);
    return vs_der_(o, 0xa3, v, n);
}

/* A new key and its self-signed certificate as DIR/vserv.key and DIR/vserv.crt: 0, or -1 (errno says why). The
 * certificate goes in first, the key last: a start that finds no key makes both again. */
static i32 vs_tls_make_(const char *kpath, const char *cpath)
{
    static const unsigned char oid_sig[] = { 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02 };          /* ecdsa-with-SHA256 */
    static const unsigned char oid_ec[] = { 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01 };                 /* id-ecPublicKey */
    static const unsigned char oid_p256[] = { 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07 };         /* prime256v1 */
    static const unsigned char oid_cn[] = { 0x55, 0x04, 0x03 };
    br_hmac_drbg_context rng;
    br_prng_seeder seed = br_prng_seeder_system((const char **)0);
    br_ec_private_key sk;
    br_ec_public_key pk;
    br_sha256_context sc;
    unsigned char kbuf[BR_EC_KBUF_PRIV_MAX_SIZE], pbuf[BR_EC_KBUF_PUB_MAX_SIZE], serial[16], h[32], sig[80];
    unsigned char a[1024], b[1024], tbs[2048], cert[2048], alg[32], name[64], *q;
    size_t n, na, nb, nalg, nname, ntbs, ncert, ns;
    char *pem, *tmp;
    i32 r = -1;
    if (!seed) { errno = ENOSYS; return -1; }
    br_hmac_drbg_init(&rng, &br_sha256_vtable, (const void *)0, 0);
    if (!seed(&rng.vtable)) { errno = EIO; return -1; }
    if (!br_ec_keygen(&rng.vtable, br_ec_get_default(), &sk, kbuf, BR_EC_secp256r1)
        || !br_ec_compute_pub(br_ec_get_default(), &pk, pbuf, &sk)) { errno = EIO; return -1; }
    br_hmac_drbg_generate(&rng, serial, sizeof(serial));
    serial[0] = (unsigned char)((serial[0] & 0x7f) | 0x40);      /* positive, 16 bytes */

    nalg = vs_der_(alg, 0x30, alg, vs_der_(alg, 0x06, oid_sig, sizeof(oid_sig)));
    n = vs_der_(a, 0x06, oid_cn, sizeof(oid_cn)); n += vs_der_(a + n, 0x0c, (const unsigned char *)"vserv", 5);
    n = vs_der_(b, 0x30, a, n); n = vs_der_(a, 0x31, b, n); nname = vs_der_(name, 0x30, a, n);
    q = tbs;
    q += vs_der_(q, 0xa0, (const unsigned char *)"\x02\x01\x02", 3);   /* v3: the extensions below */
    q += vs_der_(q, 0x02, serial, sizeof(serial));
    memcpy(q, alg, nalg); q += nalg;
    memcpy(q, name, nname); q += nname;
    na = vs_der_(a, 0x17, (const unsigned char *)"000101000000Z", 13);        /* 2000-01-01 */
    na += vs_der_(a + na, 0x18, (const unsigned char *)"99991231235959Z", 15); /* no end (RFC 5280, 4.1.2.5) */
    q += vs_der_(q, 0x30, a, na);
    memcpy(q, name, nname); q += nname;                          /* self-signed: the subject is the issuer */
    na = vs_der_(a, 0x06, oid_ec, sizeof(oid_ec)); na += vs_der_(a + na, 0x06, oid_p256, sizeof(oid_p256));
    nb = vs_der_(b, 0x30, a, na);
    a[0] = 0; memcpy(a + 1, pk.q, pk.qlen);                     /* BIT STRING: no unused bits, then the point */
    nb += vs_der_(b + nb, 0x03, a, pk.qlen + 1);
    q += vs_der_(q, 0x30, b, nb);
    q += vs_tls_exts_(q, pk.q, pk.qlen);
    ntbs = vs_der_(tbs, 0x30, tbs, (size_t)(q - tbs));

    br_sha256_init(&sc); br_sha256_update(&sc, tbs, ntbs); br_sha256_out(&sc, h);
    if (!(ns = br_ecdsa_sign_asn1_get_default()(br_ec_get_default(), &br_sha256_vtable, h, &sk, sig + 1))) { errno = EIO; goto out; }
    sig[0] = 0;
    q = cert;
    memcpy(q, tbs, ntbs); q += ntbs;
    memcpy(q, alg, nalg); q += nalg;
    q += vs_der_(q, 0x03, sig, ns + 1);
    ncert = vs_der_(cert, 0x30, cert, (size_t)(q - cert));

    if (!(tmp = (char *)malloc(strlen(kpath) + 8))) goto out;
    if ((pem = vs_pem_of_(cert, ncert, "CERTIFICATE")) != (char *)0) {
        sprintf(tmp, "%s.tmp", cpath);
        if (vs_write_new_(tmp, pem, strlen(pem), 0644) == 0 && rename(tmp, cpath) == 0) {
            free(pem);
            n = br_encode_ec_pkcs8_der((void *)0, &sk, &pk);
            q = (unsigned char *)malloc(n);
            pem = q && br_encode_ec_pkcs8_der(q, &sk, &pk) == n ? vs_pem_of_(q, n, "PRIVATE KEY") : (char *)0;
            if (q) { explicit_bzero(q, n); free(q); }
            sprintf(tmp, "%s.tmp", kpath);
            if (pem && vs_write_new_(tmp, pem, strlen(pem), 0600) == 0 && rename(tmp, kpath) == 0) { store_fsync_dir_(kpath); r = 0; }
        }
        if (pem) { explicit_bzero(pem, strlen(pem)); free(pem); }
    }
    free(tmp);
out:
    explicit_bzero(kbuf, sizeof(kbuf)); explicit_bzero(&rng, sizeof(rng));
    return r;
}

/* The key and certificate next to the store at path (a STORE.vxi, or a tenant directory): made the first time,
 * read after that. *cert and *key get their paths (malloc'd). 0, or -1 with the reason on stderr. */
static i32 vs_tls_auto_(const char *path, i32 isdir, const char **cert, const char **key)
{
    const char *sl = isdir ? (const char *)0 : strrchr(path, '/');
    size_t dl = isdir ? strlen(path) : (sl ? (size_t)(sl - path) : 0);
    char *k = (char *)malloc(dl + 16), *c = (char *)malloc(dl + 16);
    struct stat sb;
    i32 hk, hc;
    if (!k || !c) return -1;
    if (dl) { memcpy(k, path, dl); memcpy(c, path, dl); } else { k[0] = c[0] = '.'; dl = 1; }
    strcpy(k + dl, "/vserv.key"); strcpy(c + dl, "/vserv.crt");
    hk = stat(k, &sb) == 0; hc = stat(c, &sb) == 0;
    if (hk && !hc) { fprintf(stderr, "vserv: %s is there but %s is not: move the key away to make both again\n", k, c); return -1; }
    if (!hk) {
        if (vs_tls_make_(k, c) != 0) {
            fprintf(stderr, "vserv: cannot make a TLS key in %.*s: %s (give -c CERT -k KEY, or -P to serve plaintext)\n", (int)dl, k, strerror(errno));
            return -1;
        }
        fprintf(stderr, "vserv: made a TLS key and certificate: %s, %s\n", k, c);
    } else if (stat(k, &sb) == 0 && (sb.st_mode & 077)) {
        fprintf(stderr, "vserv: %s can be read by others (mode %03o): chmod 600 it\n", k, (unsigned)(sb.st_mode & 0777));
        return -1;
    }
    *cert = c; *key = k;
    return 0;
}

#endif
