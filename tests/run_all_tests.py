#!/usr/bin/env python
"""
Anatropic validation test suite runner.

Runs all three validation tests and reports pass/fail for each:
  1. Sod shock tube (exact Riemann solution)
  2. Jeans instability (unstable growth + stable oscillation)
  3. Acoustic wave propagation (phase velocity + amplitude)
"""

import os
import sys
import time

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_all():
    """Run all validation tests and report results."""
    print("=" * 60)
    print("  Anatropic Validation Test Suite")
    print("=" * 60)

    results = {}

    # --- Test 1: Sod shock tube ---
    print("\n" + "-" * 60)
    print("  Test 1: Sod Shock Tube")
    print("-" * 60)
    t_start = time.time()
    try:
        from test_sod import run_sod_test
        passed, error = run_sod_test()
        results['Sod shock tube'] = {
            'passed': passed,
            'detail': f'max density error = {error:.6f}',
            'time': time.time() - t_start,
        }
    except Exception as e:
        results['Sod shock tube'] = {
            'passed': False,
            'detail': f'EXCEPTION: {e}',
            'time': time.time() - t_start,
        }
        import traceback
        traceback.print_exc()

    # --- Test 2: Jeans instability ---
    print("\n" + "-" * 60)
    print("  Test 2: Jeans Instability")
    print("-" * 60)
    t_start = time.time()
    try:
        from test_jeans import run_jeans_test
        passed = run_jeans_test()
        results['Jeans instability'] = {
            'passed': passed,
            'detail': 'growth + oscillation tests',
            'time': time.time() - t_start,
        }
    except Exception as e:
        results['Jeans instability'] = {
            'passed': False,
            'detail': f'EXCEPTION: {e}',
            'time': time.time() - t_start,
        }
        import traceback
        traceback.print_exc()

    # --- Test 3: Acoustic wave ---
    print("\n" + "-" * 60)
    print("  Test 3: Acoustic Wave Propagation")
    print("-" * 60)
    t_start = time.time()
    try:
        from test_acoustic import run_acoustic_test
        passed, error = run_acoustic_test()
        results['Acoustic wave'] = {
            'passed': passed,
            'detail': f'phase velocity error = {error*100:.4f}%',
            'time': time.time() - t_start,
        }
    except Exception as e:
        results['Acoustic wave'] = {
            'passed': False,
            'detail': f'EXCEPTION: {e}',
            'time': time.time() - t_start,
        }
        import traceback
        traceback.print_exc()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, result in results.items():
        status = "PASS" if result['passed'] else "FAIL"
        if not result['passed']:
            all_passed = False
        print(f"  [{status}] {name}: {result['detail']} "
              f"({result['time']:.1f}s)")

    print("-" * 60)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        n_fail = sum(1 for r in results.values() if not r['passed'])
        n_total = len(results)
        print(f"  {n_fail}/{n_total} TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    all_passed = run_all()
    sys.exit(0 if all_passed else 1)
