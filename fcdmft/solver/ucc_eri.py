#! /usr/bin/env python
"""
CC impurity solver.
"""

import numpy as np
import scipy.linalg as la
import h5py

from pyscf import cc
import time
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc.uccsd import _ChemistsERIs

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    """
    Hacked CC make eri function. NOTE the order.
    """

    cput0 = (time.process_time(), time.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    else:
        # ZHC NOTE the order, aa, bb, ab
        #if(len(np.shape(mycc._scf._eri)) == 1): # ie still spin restricted
           #mycc._scf._eri = np.asarray((mycc._scf._eri, )*3)
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[1], mob), nmob)
        # ZHC NOTE FIXME
        #eri_ab = ao2mo.general(mycc._scf._eri[2], (moa,moa,mob,mob), compact=False)
        norb = moa.shape[-2]
        eri_ab = ao2mo.general(ao2mo.restore(1, mycc._scf._eri[2], norb), (moa,moa,mob,mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    if not callable(ao2mofn):
        ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
        vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
        idxa = np.tril_indices(nvira)
        idxb = np.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

        OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    return eris

cc.uccsd._make_eris_incore = _make_eris_incore
