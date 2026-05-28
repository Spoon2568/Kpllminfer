/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

.macro PROLOGUE
    .text;
    .p2align 2;
    .global gemm_ncopy_matrix;
    .type gemm_ncopy_matrix, % function;
gemm_ncopy_matrix:
.endm

.macro EPILOGUE
.endm

.macro SAVE_REGS
    .align 5
    add     sp, sp, #-(11 * 16)
    stp     d8, d9, [sp, #(0 * 16)]
    stp     d10, d11, [sp, #(1 * 16)]
    stp     d12, d13, [sp, #(2 * 16)]
    stp     d14, d15, [sp, #(3 * 16)]
    stp     d16, d17, [sp, #(4 * 16)]
    stp     x18, x19, [sp, #(5 * 16)]
    stp     x20, x21, [sp, #(6 * 16)]
    stp     x22, x23, [sp, #(7 * 16)]
    stp     x24, x25, [sp, #(8 * 16)]
    stp     x26, x27, [sp, #(9 * 16)]
    stp     x28, x29, [sp, #(10 * 16)]
.endm

.macro RESTORE_REGS
    ldp     d8, d9, [sp, #(0 * 16)]
    ldp     d10, d11, [sp, #(1 * 16)]
    ldp     d12, d13, [sp, #(2 * 16)]
    ldp     d14, d15, [sp, #(3 * 16)]
    ldp     d16, d17, [sp, #(4 * 16)]
    ldp     x18, x19, [sp, #(5 * 16)]
    ldp     x20, x21, [sp, #(6 * 16)]
    ldp     x22, x23, [sp, #(7 * 16)]
    ldp     x24, x25, [sp, #(8 * 16)]
    ldp     x26, x27, [sp, #(9 * 16)]
    ldp     x28, x29, [sp, #(10 * 16)]
    add     sp, sp, #(11*16)
.endm

#define PRFM_DIST 512
#ifdef PRFM_DIST
#define PRFM_INS(...) __VA_ARGS__
#else
#define PRFM_INS(...)
#endif

#define     M               x0
#define     N               x1
#define     A               x2
#define     LDA             x3
#define     pB              x4
#define     pA              x5
#define     stepM           x6
#define     stepMLeft       x7
#define     counter         x8
#define     stepN           x9
#define     leftN           x10
#define     stepNDiv4       x11
#define     temp            x12
#define     pAPrfm          x13
#define     NLeft           x18

#define     loadPReg        p0
#define     storePReg       p1
#define     allPReg         p3

    PROLOGUE
    SAVE_REGS
    smstart
    ptrue   allPReg.b, ALL

.LoopMStart_M1VL:
    mov     pA, A                           // pA = A
    add     A, A, LDA, LSL 4                // A += LDA * 16
    whilelt storePReg.s, xzr, M             // storePReg = svwhilelt_b32(0, M)
    cntp    stepM, storePReg, storePReg.s   // stepM = min(M, 16)

    mov     leftN, N

.LoopNStart_M1VL:
    whilelt loadPReg.b, xzr, leftN          // loadPReg = svwhilelt_b8(0, leftN)
    cntp    stepN, loadPReg, loadPReg.b     // stepN = min(leftN, 64)
    // pAPrfm = pA + (PRFM_DIST < leftN ? PRFM_DIST : 0)
    PRFM_INS(cmp    leftN, PRFM_DIST)
    PRFM_INS(mov    pAPrfm, PRFM_DIST)
    PRFM_INS(csel   pAPrfm, pAPrfm, xzr, GT)
    PRFM_INS(add    pAPrfm, pAPrfm, pA)

    // load
    mov     counter, stepM
    mov     w12, 0
.LoopLoadStart_M1VL:
    ld1b    z0.b, loadPReg/Z, [pA]
    mova    za0h.s[w12, 0], allPReg/m, z0.s
    PRFM_INS(prfm   PLDL1KEEP, [pAPrfm])
    add     pA, pA, LDA
    PRFM_INS(add    pAPrfm, pAPrfm, LDA)
    add     w12, w12, 1
    subs    counter, counter, 1
    bgt     .LoopLoadStart_M1VL

    // pA -= stepM * LDA
    mul     temp, stepM, LDA
    sub     pA, pA, temp
    PRFM_INS(sub   pAPrfm, pAPrfm, temp)
    // pA += stepN
    add     pA, pA, stepN
    PRFM_INS(add   pAPrfm, pAPrfm, stepN)

    // stepNDiv4 = stepN / 4
    lsr     stepNDiv4, stepN, 2
    // NLeft = stepN % 4
    sub     NLeft, stepN, stepNDiv4, LSL 2

    // store
    mov     counter, stepNDiv4
    mov     w12, 0
    cmp     counter, 0
    ble     .LoopStoreEnd_M1VL
.LoopStoreStart_M1VL:
    mova    z0.s, storePReg/m, za0v.s[w12, 0]
    st1w    z0.s, storePReg, [pB]
    add     pB, pB, stepM, LSL 2
    add     w12, w12, 1
    subs    counter, counter, 1
    bgt     .LoopStoreStart_M1VL
.LoopStoreEnd_M1VL:

    cmp     NLeft, 0
    bgt     .NLeft_M1VL
    b       .NLeftEnd_M1VL
.NLeft_M1VL:
    mov     x14, stepNDiv4
    mova    z0.s, allPReg/m, za0v.s[w14, 0]
    adrp    temp, .NLeftTable
    add     temp, temp, :lo12:.NLeftTable
    add     temp, temp, NLeft, LSL 6
    ld1b    z2.b, allPReg/z, [temp]
    tbl     z0.b, z0.b, z2.b
    mul     temp, stepM, NLeft
    whilelt p4.b, xzr, temp
    st1b    z0.b, p4, [pB]
    add     pB, pB, temp
.NLeftEnd_M1VL:

    // leftN -= stepN
    subs    leftN, leftN, stepN
    bgt     .LoopNStart_M1VL

    subs    M, M, 16
    bgt     .LoopMStart_M1VL

    smstop
    RESTORE_REGS
    ret

.align 3
.NLeftTable:
    .xword  0, 0, 0, 0, 0, 0, 0, 0
    .xword  0x1c1814100c080400, 0x3c3834302c282420, 0, 0, 0, 0, 0, 0
    .xword  0x0d0c090805040100, 0x1d1c191815141110, 0x2d2c292825242120, 0x3d3c393835343130, 0, 0, 0, 0
    .xword  0x0908060504020100, 0x141211100e0d0c0a, 0x1e1d1c1a19181615, 0x2928262524222120, 0x343231302e2d2c2a, \
            0x3e3d3c3a39383635, 0, 0

    EPILOGUE
