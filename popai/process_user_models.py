import os
import numpy as np
import logging
import msprime
import configparser
import io
import pandas as pd
import ast
import demesdraw # ModelBuilder
import matplotlib.pyplot as plt # ModelBuilder
from matplotlib.backends.backend_pdf import PdfPages
import copy
import re

class ModelReader:

    """Read user specified models."""

    def __init__(self, config_values):
        self.config = config_values
        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def read_models(self):

        # find the .model files. Each name ends in _<number>.model, and the model's label is
        # model_<number>, which later steps use to find each model's simulations.
        model_files = {}
        for name in os.listdir(self.config["user models"]):
            if not name.endswith('.model'):
                continue
            match = re.search(r"_(\d+)\.model$", name)
            if not match:
                raise ValueError(f"The model file {name} must end in _x.model, where x is a whole "
                                 f"number (for example, model_0.model).")
            number = int(match.group(1))
            if number in model_files:
                raise ValueError(f"The model files {model_files[number]} and {name} both end in "
                                 f"_{number}. Each model needs its own number.")
            model_files[number] = name
        if not model_files:
            raise ValueError(f"There are no .model files in {self.config['user models']}.")

        # iterate over model files
        demographies = []
        labels = []
        for number in sorted(model_files):
            model = model_files[number]

            # list for storing active populations and populations with migration at the present.
            active_populations = []
            pops_w_present_migration = []

            modelinfo = configparser.ConfigParser(inline_comment_prefixes="#")
            modelinfo.optionxform = str
            modelinfo.read(os.path.join(self.config["user models"],model))
            if not modelinfo.has_section("Populations"):
                raise ValueError(f"Error in {model}: there is no [Populations] section.")
            if not modelinfo.has_section("Events"):
                modelinfo.add_section("Events")   # a model can have no events (one population)

            # migration matrix, if the model has one (read once; rates drawn per replicate)
            migration_matrix = self._read_migration_matrix(modelinfo, model)

            # read the events once, and order them so each comes after the events it depends on
            events = {name: self._split_parameters(text) for name, text in modelinfo["Events"].items()}
            event_order = self._order_events(events, model)
            self._check_numbers(modelinfo, events, model)

            # iterate over parameterizations
            for _ in range(self.config["replicates"]):

                # dictionary for storing event names and times
                event_dict = {}

                # empty demography
                demography = msprime.Demography()
                splits = []   # (time, descendants, ancestor) for each split in this replicate
                # iterate over populations and add to demogrpahy
                for item in modelinfo["Populations"]:
                    initial_size = self._population_size(item, modelinfo["Populations"][item], event_dict, model)
                    demography.add_population(name = item, initial_size=initial_size)
                    active_populations.append(item)
                    event_dict[f"pop_{item}"] = initial_size

                for item in event_order:

                    item_dict = events[item]

                    if item_dict['event'] == 'split':

                        if not all(key in item_dict for key in ['time', 'descendants', 'ancestor']):
                            raise Exception(f"Check your split events. All must include a time, descendants, and an ancestor.")

                        event_time = self._split_time(item, item_dict, splits, event_dict, model)
                        demography.add_population_split(derived=item_dict['descendants'], ancestral=item_dict['ancestor'], time=event_time)
                        splits.append((event_time, item_dict['descendants'], item_dict['ancestor']))
                        active_populations = [x for x in active_populations if x != item_dict['ancestor']]
                        event_dict[item] = event_time

                    elif item_dict['event'] == 'symmetric_migration':

                        if not all(key in item_dict for key in ['start', 'stop', 'populations', 'rate']):
                            raise Exception(f"Check your symmetric migration events. All must include a start and stop time, populations, and a rate.")

                        start_event_time, stop_event_time = self._migration_window(item, item_dict, item_dict['populations'], splits, event_dict, model)
                        
                        # one draw for both directions; as Nm, each direction is divided by its own source's size
                        rate, is_nm = self._migration_rate(item, item_dict, event_dict, model)
                        population_0, population_1 = item_dict['populations']
                        migration_rate_0 = rate / event_dict[f"pop_{population_1}"] if is_nm else rate
                        migration_rate_1 = rate / event_dict[f"pop_{population_0}"] if is_nm else rate\

                        demography.add_migration_rate_change(source=item_dict['populations'][1], dest=item_dict['populations'][0], time=start_event_time, rate=migration_rate_0)
                        demography.add_migration_rate_change(source=item_dict['populations'][0], dest=item_dict['populations'][1], time=start_event_time, rate=migration_rate_1)

                        demography.add_migration_rate_change(source=item_dict['populations'][1], dest=item_dict['populations'][0], time=stop_event_time, rate=0)
                        demography.add_migration_rate_change(source=item_dict['populations'][0], dest=item_dict['populations'][1], time=stop_event_time, rate=0)


                        event_dict[item] = [start_event_time, stop_event_time]

                    elif item_dict['event'] == 'asymmetric_migration':

                        if not all(key in item_dict for key in ['start', 'stop', 'source', 'dest', 'rate']):
                            raise Exception(f"Check your assymmetric migration events. All must include a start and stop time, a source, a dest, and a rate.")


                        start_event_time, stop_event_time = self._migration_window(item, item_dict, [item_dict['source'], item_dict['dest']], splits, event_dict, model)
                        rate, is_nm = self._migration_rate(item, item_dict, event_dict, model)
                        migration_rate = rate / event_dict[f"pop_{item_dict['source']}"] if is_nm else rate

                        demography.add_migration_rate_change(source=item_dict['source'], dest=item_dict['dest'], time=start_event_time, rate=migration_rate)
                        demography.add_migration_rate_change(source=item_dict['source'], dest=item_dict['dest'], time=stop_event_time, rate=0)

                        event_dict[item] = [start_event_time, stop_event_time]

                    elif item_dict['event'] == 'popsize':

                        if not all(key in item_dict for key in ['time', 'size', 'population']):
                            raise Exception(f"Check your pop size events. All must include a time, a population, and a size.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        population_size = self._get_event_value(item_dict, event_dict, 'size')

                        demography.add_population_parameters_change(population=item_dict['population'], time=event_time, initial_size=population_size, growth_rate=0)
                        event_dict[item] = event_time

                    elif item_dict['event'] == 'popgrowth':

                        if not all(key in item_dict for key in ['time', 'rate', 'population']):
                            raise Exception(f"Check your pop growth events. All must include a time, a population, and a rate.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        growth_rate = self._get_event_value(item_dict, event_dict, 'rate')
                    
                        demography.add_population_parameters_change(population=item_dict['population'], time=event_time, initial_size=None, growth_rate=growth_rate)

                        event_dict[item] = event_time

                    elif item_dict['event'] == 'bottleneck':

                        if not all(key in item_dict for key in ['time', 'population', 'prop']):
                            raise Exception(f"Check your bottleneck events. All must include a time, a population, and a prop.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        bottleneck_prop = self._get_event_value(item_dict, event_dict, 'prop')

                        demography.add_simple_bottleneck(population=item_dict['population'], time=event_time, proportion=bottleneck_prop)
                        
                        event_dict[item] = event_time

                    elif item_dict['event'] == 'pulse':

                        if not all(key in item_dict for key in ['time', 'source', 'dest', 'prop']):
                            raise Exception(f"Check your pulse events. All must include a time, a source, a dest, and a prop.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        pulse_prop = self._get_event_value(item_dict, event_dict, 'prop')

                        demography.add_mass_migration(source=item_dict['source'], dest=item_dict['dest'], time=event_time, proportion=pulse_prop)
                        
                        event_dict[item] = event_time

                    else:
                        raise Exception(f"Type {item_dict['event']} is not a valid option. Valid options include split, symmetric migration, asymmetric migration, popsize, popgrowth, and bottleneck.")

                if migration_matrix is not None:
                    self._add_matrix_migration(demography, migration_matrix, splits, event_dict, model)
                demography.sort_events()
                demographies.append(demography)
                labels.append(f"model_{number}")
        return(demographies, labels)

    def _check_numbers(self, modelinfo, events, model):
        """Before drawing anything, check the numbers written in the model file: a range must
        have its smaller value first, and values must make sense (times and migration rates
        can't be negative, sizes must be above 0, proportions must be between 0 and 1).
        Expressions (like 2*div_AB) aren't known until they're drawn, so they're skipped."""

        # (what, event type, key, ends) for every value to check
        values = [(f"population {name}", None, 'size', text.strip().strip('[]').split(','))
                  for name, text in modelinfo["Populations"].items()]
        for name, item in events.items():
            for key in ('time', 'start', 'stop', 'size', 'prop', 'rate'):
                if key in item:
                    ends = item[key] if isinstance(item[key], list) else [item[key]]
                    values.append((f"event {name}", item.get('event'), key, ends))

        for what, event_type, key, ends in values:
            try:
                numbers = [float(re.sub(r'Nm$', '', end.strip())) for end in ends]   # Nm cut off the end
            except ValueError:
                continue   # an expression (or a mistake that's reported when it's drawn)
            shown = f"[{','.join(end.strip() for end in ends)}]" if len(ends) > 1 else ends[0].strip()
            problem = f"Error in {model}: the {key} of {what} is {shown}"

            if len(numbers) == 2 and numbers[0] > numbers[1]:
                raise ValueError(f"{problem}. Put the smaller value first.")
            if key == 'size' and min(numbers) <= 0:
                raise ValueError(f"{problem}, but sizes must be above 0.")
            if key in ('time', 'start', 'stop') and min(numbers) < 0:
                raise ValueError(f"{problem}, but times can't be negative.")
            if key == 'prop' and (min(numbers) < 0 or max(numbers) > 1):
                raise ValueError(f"{problem}, but proportions must be between 0 and 1.")
            if key == 'rate' and event_type != 'popgrowth' and min(numbers) < 0:
                raise ValueError(f"{problem}, but migration rates can't be negative.")

    def _order_events(self, events, model):
        """Order events so that each one comes after every event it depends on:
        - events named in its values (stop=div_AB needs div_AB first),
        - for a split, the splits that form its descendants (AB's split before ABC's),
        - for migration, the splits that form and end its populations.
        Events that don't depend on each other keep their order in the file."""

        # which split forms each population (as the ancestor), and which ends it
        formed_by, ended_by = {}, {}
        for name, item in events.items():
            if item.get('event') == 'split':
                formed_by[item.get('ancestor')] = name
                for population in item.get('descendants', []):
                    ended_by[population] = name

        depends_on = {}
        for name, item in events.items():
            needs = set()
            # events named in values that can hold expressions
            for key in ('time', 'start', 'stop', 'size', 'rate', 'prop'):
                values = item.get(key, [])
                for value in (values if isinstance(values, list) else [values]):
                    needs.update(word for word in re.findall(r'[A-Za-z_]\w*', value) if word in events)
            if item.get('event') == 'split':
                needs.update(formed_by[d] for d in item.get('descendants', []) if d in formed_by)
            elif item.get('event') in ('symmetric_migration', 'asymmetric_migration'):
                populations = item.get('populations') or [item.get('source'), item.get('dest')]
                for population in populations:
                    needs.update(split for split in (formed_by.get(population), ended_by.get(population)) if split)
            depends_on[name] = needs

        # repeatedly take the first event (in file order) whose dependencies are all done
        order, remaining = [], list(events)
        while remaining:
            ready = [name for name in remaining if depends_on[name] <= set(order)]
            if not ready:
                raise ValueError(
                    f"Error in {model}: the events {', '.join(remaining)} can't be put in order, "
                    f"because some of them depend on each other in a loop (for example, an event "
                    f"whose time refers to a second event whose time refers back to the first).")
            order.append(ready[0])
            remaining.remove(ready[0])
        return order

    def _migration_window(self, item, item_dict, populations, splits, event_dict, model):
        """Draw the start and stop times of a migration event (backwards in time) so that
        - the stop comes after the start, and
        - the start falls while the populations all exist: after the split that forms
          them (if they are ancestral) and before the split that ends them.
        Raises an error only if the model's values make that impossible."""

        # when the populations all exist, in this replicate
        exists_from, exists_until = 0, np.inf
        for time, descendants, ancestor in splits:
            for population in populations:
                if population == ancestor:
                    exists_from = max(exists_from, time)
                if population in descendants:
                    exists_until = min(exists_until, time)

        start_min, start_max = self._value_range(item_dict, event_dict, 'start')
        stop_min, stop_max = self._value_range(item_dict, event_dict, 'stop')

        # start: in its own range, while the populations exist, and before the latest possible stop
        # (times are whole numbers, so "before" means at least one generation before)
        low = max(start_min, exists_from)
        high = min(start_max, exists_until - 1, stop_max - 1)
        if low > high:
            raise ValueError(
                f"Error in {model}: no start time works for {item}. The start must be between "
                f"{start_min:g} and {start_max:g}, before the stop (at most {stop_max:g}), and while "
                f"{' and '.join(populations)} all exist (from {exists_from:g} to {exists_until:g} "
                f"generations ago in this replicate).")
        start = self._draw_time(low, high)

        # stop: in its own range, and after the start
        stop = self._draw_time(max(stop_min, start + 1), stop_max)
        return start, stop

    def _split_time(self, item, item_dict, splits, event_dict, model):
        """Draw the time of a split so it is no younger than the splits that form its
        descendants (AB must form before it can split from C, backwards in time).
        Raises an error only if the model's values make that impossible."""

        # times of the splits (already drawn) that form this split's descendants
        formed = [time for time, _, ancestor in splits if ancestor in item_dict['descendants']]

        time_min, time_max = self._value_range(item_dict, event_dict, 'time')

        # in its own range, and no younger than each of those splits (the same time is fine:
        # msprime applies splits at the same time in the order they were added)
        low = max([time_min] + formed)
        if low > time_max:
            raise ValueError(
                f"Error in {model}: no time works for {item}. Its time must be between "
                f"{time_min:g} and {time_max:g}, and no younger than the splits that form its "
                f"descendants (at {', '.join(f'{t:g}' for t in formed) or 'none'} "
                f"generations ago in this replicate).")
        return self._draw_time(low, time_max)

    def _population_size(self, population, value, event_dict, model):
        """Draw a population's size. The size can be a range ([10000,50000]), a number
        (20000), the name of a population listed above it (the same size as that population),
        or an expression using those sizes (0.5*pop_A)."""
        value = value.strip()
        if f"pop_{value}" in event_dict:
            return event_dict[f"pop_{value}"]
        try:
            if value.startswith('['):
                ends = [end.strip() for end in value.strip('[]').split(',')]
                if len(ends) != 2:
                    raise ValueError
                low, high = [self._evaluate_var(end, event_dict) for end in ends]
                return self._draw_time(low, high)
            return self._evaluate_var(value, event_dict)
        except (ValueError, SyntaxError, TypeError):
            raise ValueError(f"Error in {model}: can't read the size of population {population}, "
                             f"'{value}'. Use a range ([10000,50000]), a number (20000), or the "
                             f"name of a population listed above it.") from None

    def _migration_rate(self, item, item_dict, event_dict, model):
        """Draw the rate of a migration event. The rate can be fixed (1e-5), a range
        ([1e-5,1e-4]), or a number of migrants (1Nm or [0.5Nm,2Nm]).
        Returns (rate, is_nm); an Nm value still needs dividing by the source's size."""

        value = item_dict['rate']
        ends = [end.strip() for end in (value if isinstance(value, list) else [value, value])]
        shown = f"[{','.join(value)}]" if isinstance(value, list) else value   # for error messages

        for end in ends:
            if "Nm" in end and not end.endswith("Nm"):
                raise ValueError(f"Error in {model}: the rate of {item} is {shown}. "
                                 f"Write Nm after the number, as in 1Nm or [0.5Nm,2Nm].")
        is_nm = [end.endswith("Nm") for end in ends]
        if is_nm[0] != is_nm[1]:
            raise ValueError(f"Error in {model}: the rate of {item} is {shown}. Both ends of "
                             f"a range must be rates, or both Nm values.")
        if is_nm[0]:
            ends = [end[:-2] for end in ends]

        low, high = [self._evaluate_var(end, event_dict) for end in ends]
        rate = low if low == high else self.rng.uniform(low=low, high=high, size=1)[0]
        return rate, is_nm[0]

    def _value_range(self, item_dict, event_dict, key):
        """The smallest and largest values a time can take: the two ends of a [min,max]
        range, or the same number twice for a fixed value or an expression."""
        value = item_dict[key]
        if isinstance(value, list):
            return (self._evaluate_var(value[0], event_dict),
                    self._evaluate_var(value[1], event_dict))
        value = self._evaluate_var(value, event_dict)
        return value, value

    def _draw_time(self, low, high):
        """A whole-number time drawn uniformly between low and high (or low, if they're equal)."""
        if low == high:
            return low
        return np.round(self.rng.uniform(low=low, high=high, size=1)[0], 0)

    def _read_migration_matrix(self, modelinfo, model):
        """Read the [Migration] section of a model file, if there is one.

        The section holds one table, with population names as column headers and as the
        first entry of each row. Each cell is 0 (no migration), a rate (1e-5), a uniform
        prior on the rate ([1e-5,1e-4]), or the same as a number of migrants (Nm1 or
        [0.5 Nm,2 Nm]). The cell in row j, column k is msprime's M[j, k]: the rate at which
        lineages move from population j to population k, backwards in time. Each cell is
        drawn separately.

        Returns {(source, dest): (low, high, is_nm)} for the non-zero cells, or None."""

        if not modelinfo.has_section("Migration"):
            return None

        populations = list(modelinfo["Populations"])
        keys = list(modelinfo["Migration"])
        if len(keys) != 1:
            raise ValueError(f"Error in {model}: the [Migration] section should hold one table "
                             f"(for example 'matrix = ...'), but it has {len(keys)} entries.")

        # split each line into cells, keeping [min, max] together even if it contains spaces
        lines = [re.findall(r"\[[^\]]*\]|\S+(?:\s+Nm\b)?", line)
                 for line in modelinfo["Migration"][keys[0]].splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Error in {model}: the [Migration] table is empty.")
        header, rows = lines[0], lines[1:]

        row_names = [row[0] for row in rows]
        if sorted(row_names) != sorted(header):
            raise ValueError(f"Error in {model}: the [Migration] table should have the same "
                             f"populations as rows and columns, but has columns {header} "
                             f"and rows {row_names}.")
        unknown = [name for name in header if name not in populations]
        if unknown:
            raise ValueError(f"Error in {model}: the [Migration] table includes {unknown}, "
                             f"which are not listed in [Populations].")

        cells = {}
        for row in rows:
            source, values = row[0], row[1:]
            if len(values) != len(header):
                raise ValueError(f"Error in {model}: row {source} of the [Migration] table has "
                                 f"{len(values)} entries, but there are {len(header)} columns.")
            for dest, value in zip(header, values):
                rate = self._parse_matrix_cell(value, source, dest, model)
                if rate is None:
                    continue
                if source == dest:
                    raise ValueError(f"Error in {model}: the [Migration] table has migration "
                                     f"from {source} to itself. Diagonal entries must be 0.")
                cells[(source, dest)] = rate
        return cells

    def _parse_matrix_cell(self, value, source, dest, model):
        """One cell of the migration matrix: None for no migration, else (low, high, is_nm)."""
        ends = [x.strip() for x in value.strip("[]").split(",")]
        if len(ends) == 1:
            ends = ends * 2
        problem = (f"Error in {model}: the [Migration] entry for {source} to {dest} is '{value}'. "
                   "Use 0, a rate (1e-5), a range ([1e-5,1e-4]), or Nm values (Nm1 or [Nm0.5,Nm2]).")
        if len(ends) != 2:
            raise ValueError(problem)
        is_nm = [end.endswith("Nm") for end in ends]
        if is_nm[0] != is_nm[1]:
            raise ValueError(problem + " Both ends of a range must be rates, or both Nm values.")
        try:
            low, high = [float(end[:-2] if is_nm[0] else end) for end in ends]
        except ValueError:
            raise ValueError(problem) from None
        if low < 0 or low > high:
            raise ValueError(problem + " Values must be non-negative, minimum first.")
        if high == 0:
            return None
        return (low, high, is_nm[0])

    def _add_matrix_migration(self, demography, cells, splits, event_dict, model):
        """Set migration from the present, as given by the migration matrix. msprime turns
        it off when the populations involved split from their ancestor."""

        merged_at_present = {d for time, descendants, _ in splits if time == 0 for d in descendants}
        ancestral = {ancestor for time, _, ancestor in splits if time > 0}

        for (source, dest), (low, high, is_nm) in cells.items():
            for population in (source, dest):
                if population in ancestral:
                    raise ValueError(
                        f"Error in {model}: the migration matrix includes migration between "
                        f"{source} and {dest}, but {population} is an ancestral population (it "
                        f"forms at a split before the present). The matrix sets migration from "
                        f"the present, so it can only include present-day populations. For "
                        f"migration involving ancestral populations, use a symmetric_migration "
                        f"or asymmetric_migration event with start and stop times.")
                if population in merged_at_present:
                    raise ValueError(
                        f"Error in {model}: the migration matrix includes migration between "
                        f"{source} and {dest}, but {population} merges with another population "
                        f"at time 0, so it doesn't exist at the present. Use the name of the "
                        f"merged population instead.")
            rate = low if low == high else self.rng.uniform(low=low, high=high)
            if is_nm:
                # M[j, k] is a fraction of population j (forward in time), so m = Nm / N_j
                rate = rate / event_dict[f"pop_{source}"]
            demography.set_migration_rate(source=source, dest=dest, rate=rate)

    def validate_models(self, demographies, labels, outplot=None):
        """
        Plot example models demographies.

        Parameters:
            demographies (List): demographies
            labels (List): model labels
            outplot (string): path to store output figures. Default is to show.

        Returns:
            Nothing

        Raises:
            Error: If models cannot be plotted.
        """

        try:
            self._plot_models(demographies, labels, outplot)


        except ValueError as ve:
            raise ValueError(f"ValueError: Issue when plotting example \
                             msprime demographies: {ve}") from ve

        except Exception as e:
            raise RuntimeError(f"Unexpected Error: Issue when plotting \
                               example msprime demographies: {e}") from e


    def _nonzero(self, model):
        count=0
        for event in model.events:
            if event.time == 0:
                count+=1
                event.time = count*1
        return model

    def _plot_models(self, demographies, labels, outplot):
        """Plot example models for a given type of demography."""

        if outplot is None:
            for modelix, model in enumerate(demographies):
                if modelix % self.config['replicates'] == 0:
                    new_model = copy.deepcopy(model)
                    new_model = self._nonzero(new_model)
                    try:
                        graph = new_model.to_demes()

                        # Plot the model
                        fig = plt.subplots()
                        demesdraw.tubes(graph, ax=fig[1], seed=1)
                    except:
                        logging.warning(f"There was an issue with plotting your model. Please verify that your model is working as desired. This could arise if the migration matrix includes migration between populations not extant at the present. We recommend using events for such migration, rather than the matrix. This prevent issues with plotting.")
                    plt.title(f"Model: {labels[modelix]}")
                    plt.show()

        else:
            with PdfPages(outplot) as pdf:
                for modelix, model in enumerate(demographies):
                    if modelix % self.config['replicates'] == 0:
                        new_model = copy.deepcopy(model)
                        new_model = self._nonzero(new_model)
                        try:
                            graph = new_model.to_demes()

                            # Plot the model
                            fig, ax = plt.subplots()
                            demesdraw.tubes(graph, ax=ax, seed=1)
                        except:
                            # Clear the plot
                            ax.clear()

                            # Add the warning text in the center of the blank page
                            ax.text(0.5, 0.5, "There was an issue with plotting your model. Please verify that your model is working as desired.\nThis could arise if the migration matrix includes migration between populations not extant at the present.\nWe recommend using events for such migration, rather than the matrix. This will prevent issues with plotting.",
                                    ha='center', va='center', wrap=True, fontsize=12)

                            # Hide axes
                            ax.set_xticks([])
                            ax.set_yticks([])
                            logging.warning(f"There was an issue with plotting your model. Please verify that your model is working as desired. This could arise if the migration matrix includes migration between populations not extant at the present. We recommend using events for such migration, rather than the matrix. This prevent issues with plotting.")

                        plt.title(f"Model: {labels[modelix]}")
                        pdf.savefig(fig)
                        plt.close(fig)

    def _split_parameters(self, params):

        split_results = {}

        # get event type
        event_type, params = params.split('{')
        split_results['event'] = event_type

       # split parameters by commas, ignoring commas inside brackets.
        
        result = self._split_ignore_char_extend(params)
        
        for item in result:
            if '[' in item.split('=')[1]:
                value = [x.strip('[] ') for x in self._split_ignore_char(item.split('=')[1], '(', ')')]
            else:
                value = item.split('=')[1].strip()
            split_results[item.split('=')[0].strip()] = value

        return split_results
    
    def _evaluate_var(self, var, event_dict):
        """Turn a value from a model file into a number. It can be a number (1e-5) or an
        expression using earlier events, population sizes, and min, max, mean, or median
        (for example, 0.5*pop_C or max(div_AB, 45000))."""
        try:
            return float(var)
        except ValueError:
            pass

        # the names an expression can use
        names = {'min': min, 'max': max,
                 'mean': lambda *values: float(np.mean(values)),
                 'median': lambda *values: float(np.median(values))}
        names.update(event_dict)
        try:
            return eval(var, {'__builtins__': {}}, names)
        except NameError as error:
            raise ValueError(f"Can't work out the value '{var}' ({error}). Expressions can use "
                             f"the names of earlier events, population sizes (as pop_A), and "
                             f"min, max, mean, and median.") from None

    def _split_ignore_char(self, string, char1, char2):
        result = []
        current = []
        in_brackets = False

        for char in string:
            if char == char1:
                in_brackets = True
            elif char == char2:
                in_brackets = False
            elif char == "," and not in_brackets:
                result.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        # Add the last parameter
        if current:
            result.append("".join(current).strip('}'))
        
        return(result)

    def _split_ignore_char_extend(self, string):
        result = []
        current = []
        in_brackets = False
        in_paren = False

        for char in string:
            if char == '[':
                in_brackets = True
            elif char == ']':
                in_brackets = False
            elif char == '(':
                in_paren = True
            elif char == ')':
                in_paren = False
            elif char == "," and not in_brackets and not in_paren:
                result.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        # Add the last parameter
        if current:
            result.append("".join(current).strip('}'))
        
        return(result)

    def _get_event_value(self, item_dict, event_dict, valuetype):
        """Draw a time, size, growth rate, or proportion. A fixed value or expression is used
        as it is; a [min,max] range is drawn uniformly, rounded for times and sizes."""
        low, high = self._value_range(item_dict, event_dict, valuetype)
        if valuetype in ('time', 'size'):
            return self._draw_time(low, high)
        return low if low == high else self.rng.uniform(low=low, high=high, size=1)[0]
