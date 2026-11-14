/* vserv_pem.c - TLS certificates and private keys from PEM files, decoded with BearSSL.
 * Part of vserv.c, which includes it at this point; not compiled on its own. */

#ifdef VSERV_TLS
/* ---- certificates and keys (PEM), decoded with BearSSL ---- */
typedef struct { unsigned char *p; size_t n, cap; } VsBuf;

static void vs_buf_add_(void *ctx, const void *d, size_t n)
{
    VsBuf *b = (VsBuf *)ctx;
    if (b->n + n > b->cap) { b->cap = (b->n + n) * 2 + 256; b->p = (unsigned char *)realloc(b->p, b->cap); }
    memcpy(b->p + b->n, d, n);
    b->n += n;
}

/* DER of every PEM object in the file whose name contains `want` ("CERTIFICATE", "PRIVATE KEY"); -1 on error */
static i32 vs_pem_(const char *path, const char *want, VsBuf **objs, i32 *nobj)
{
    FILE *f = fopen(path, "rb");
    VsBuf file, cur;
    br_pem_decoder_context pc;
    unsigned char chunk[4096], *p;
    size_t len, k;
    i32 in = 0, nl = 0;
    if (!f) return -1;
    memset(&file, 0, sizeof(file));
    while ((k = fread(chunk, 1, sizeof(chunk), f)) > 0) vs_buf_add_(&file, chunk, k);
    fclose(f);
    *objs = (VsBuf *)0; *nobj = 0;
    memset(&cur, 0, sizeof(cur));
    br_pem_decoder_init(&pc);
    p = file.p; len = file.n;
    for (;;) {
        if (len == 0 && !nl) { p = (unsigned char *)"\n"; len = 1; nl = 1; }   /* the last line needs its newline */
        if (len == 0) break;
        k = br_pem_decoder_push(&pc, p, len);
        p += k; len -= k;
        switch (br_pem_decoder_event(&pc)) {
        case BR_PEM_BEGIN_OBJ:
            in = strstr(br_pem_decoder_name(&pc), want) != (char *)0;
            memset(&cur, 0, sizeof(cur));
            br_pem_decoder_setdest(&pc, in ? vs_buf_add_ : (void (*)(void *, const void *, size_t))0, &cur);
            break;
        case BR_PEM_END_OBJ:
            if (in) { *objs = (VsBuf *)realloc(*objs, (size_t)(*nobj + 1) * sizeof(VsBuf)); (*objs)[(*nobj)++] = cur; }
            in = 0;
            break;
        case BR_PEM_ERROR:
            free(file.p);
            return -1;
        default:
            break;
        }
    }
    explicit_bzero(file.p, file.n);              /* the file may hold a private key */
    free(file.p);
    return 0;
}

static i32 vs_tls_load_(const char *cert, const char *key, const char *ca)
{
    VsBuf *o;
    i32 n, i;
    br_x509_decoder_context dc;
    if (vs_pem_(cert, "CERTIFICATE", &o, &n) || n == 0) { fprintf(stderr, "vserv: no certificate in %s\n", cert); return -1; }
    vs_chain = (br_x509_certificate *)calloc((size_t)n, sizeof(br_x509_certificate));
    for (i = 0; i < n; ++i) { vs_chain[i].data = o[i].p; vs_chain[i].data_len = o[i].n; }
    vs_nchain = (size_t)n;
    free(o);
    br_x509_decoder_init(&dc, (void (*)(void *, const void *, size_t))0, (void *)0);   /* who signed our certificate */
    br_x509_decoder_push(&dc, vs_chain[0].data, vs_chain[0].data_len);
    vs_issuer = (i32)br_x509_decoder_get_signer_key_type(&dc);
    if (vs_issuer == 0) vs_issuer = BR_KEYTYPE_EC;
    if (vs_pem_(key, "PRIVATE KEY", &o, &n) || n == 0) { fprintf(stderr, "vserv: no private key in %s\n", key); return -1; }
    br_skey_decoder_init(&vs_skey);
    br_skey_decoder_push(&vs_skey, o[0].p, o[0].n);
    for (i = 0; i < n; ++i) { explicit_bzero(o[i].p, o[i].n); free(o[i].p); }
    free(o);
    if (br_skey_decoder_last_error(&vs_skey) != 0) { fprintf(stderr, "vserv: cannot decode the key in %s\n", key); return -1; }
    vs_keytype = br_skey_decoder_key_type(&vs_skey);
    if (ca) {                                    /* trust anchors for client certificates */
        if (vs_pem_(ca, "CERTIFICATE", &o, &n) || n == 0) { fprintf(stderr, "vserv: no certificate in %s\n", ca); return -1; }
        vs_tas = (br_x509_trust_anchor *)calloc((size_t)n, sizeof(br_x509_trust_anchor));
        for (i = 0; i < n; ++i) {
            VsBuf dn;
            br_x509_pkey *pk;
            br_x509_trust_anchor *ta = &vs_tas[i];
            memset(&dn, 0, sizeof(dn));
            br_x509_decoder_init(&dc, vs_buf_add_, &dn);
            br_x509_decoder_push(&dc, o[i].p, o[i].n);
            pk = br_x509_decoder_get_pkey(&dc);
            if (!pk) { fprintf(stderr, "vserv: cannot decode certificate %d of %s\n", i + 1, ca); return -1; }
            ta->dn.data = dn.p; ta->dn.len = dn.n;
            ta->flags = br_x509_decoder_isCA(&dc) ? BR_X509_TA_CA : 0;   /* a non-CA anchor trusts that one certificate */
            ta->pkey.key_type = pk->key_type;
            if (pk->key_type == BR_KEYTYPE_RSA) {
                ta->pkey.key.rsa.n = (unsigned char *)malloc(pk->key.rsa.nlen); memcpy(ta->pkey.key.rsa.n, pk->key.rsa.n, pk->key.rsa.nlen);
                ta->pkey.key.rsa.nlen = pk->key.rsa.nlen;
                ta->pkey.key.rsa.e = (unsigned char *)malloc(pk->key.rsa.elen); memcpy(ta->pkey.key.rsa.e, pk->key.rsa.e, pk->key.rsa.elen);
                ta->pkey.key.rsa.elen = pk->key.rsa.elen;
            } else {
                ta->pkey.key.ec.curve = pk->key.ec.curve;
                ta->pkey.key.ec.q = (unsigned char *)malloc(pk->key.ec.qlen); memcpy(ta->pkey.key.ec.q, pk->key.ec.q, pk->key.ec.qlen);
                ta->pkey.key.ec.qlen = pk->key.ec.qlen;
            }
            free(o[i].p);
        }
        free(o);
        vs_ntas = (size_t)n;
    }
    return 0;
}
#endif
