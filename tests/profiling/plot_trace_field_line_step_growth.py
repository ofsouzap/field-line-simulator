from matplotlib import pyplot as plt
from timeit import timeit
import numpy as np
from field import Field
from point_source import PointSource


clip_ranges = np.array([
    [0, 100],
    [0, 100]
])

strength_range = (0.0, 20.0)


def single_test(test_count: int, source_count: int, pos_count: int, repeat_count: int = 10) -> np.ndarray:

    results = np.zeros(shape=(test_count,), dtype=float)

    for test_i in range(test_count):

        # Generate random field

        sources = np.random.rand(source_count, 3)

        sources[:, 0] *= clip_ranges[0, 1] - clip_ranges[0, 0]
        sources[:, 0] += clip_ranges[0, 0]
        sources[:, 1] *= clip_ranges[1, 1] - clip_ranges[1, 0]
        sources[:, 1] += clip_ranges[1, 0]
        sources[:, 1] *= strength_range[1] - strength_range[0]
        sources[:, 1] += strength_range[0]

        field = Field()

        for source_i in range(sources.shape[0]):
            field.add_element(
                PointSource(np.array([
                    sources[source_i, 0],
                    sources[source_i, 1]
                ]),
                sources[source_i, 2])
            )

        # Generate random inputs

        poss = np.random.rand(pos_count, 2)
        poss[:, 0] *= clip_ranges[0, 1] - clip_ranges[0, 0]
        poss[:, 0] += clip_ranges[0, 0]
        poss[:, 1] *= clip_ranges[1, 1] - clip_ranges[1, 0]
        poss[:, 1] += clip_ranges[1, 0]

        positives = np.random.randint(2, size=(pos_count,)) == 0

        # Put inputs in required format for running function

        lines = np.zeros(shape=(pos_count, 2, 2))
        lines[:, 0, :] = poss

        active_mask = np.ones(shape=(pos_count,), dtype=bool)

        # Perform test

        result = timeit(
            lambda: field._Field__field_line_trace_single_iteration(  # type: ignore
                t=0,
                lines=lines,
                active_mask=active_mask,
                positives=positives,
                step_distance=1,
                element_stop_distance=1,
                clip_ranges=clip_ranges
            ),
            number=repeat_count
        )

        # Store result

        results[test_i] = result

    return results


def main():

    # Generate data

    ns = np.arange(0, 20, dtype=int)

    results_list = []

    for n in ns:

        test_ress = single_test(
            test_count=50,
            source_count=n,
            pos_count=48
        )

        results_list.append((n, test_ress))

    x = np.array([n for (n, _) in results_list])

    all_y = np.array([ress for (_, ress) in results_list])
    y = np.mean(all_y, axis=1)
    yerr = np.array([
        y - np.min(all_y, axis=1),
        np.max(all_y, axis=1) - y,
    ])

    # Plot data

    _, ax = plt.subplots(1, 1)

    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr
    )

    ax.set_ylabel("Single Line Trace Iteration Running Time (seconds)")
    ax.set_xlabel("Number of Sources")

    plt.show()


if __name__ == "__main__":
    main()
