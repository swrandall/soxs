.. _changelog:

ChangeLog
=========

Version 0.4.0
-------------

This version contains new features and bugfixes. Some changes are not backwards-compatible. 

* SOXS has been re-branded as "Simulating Observations of X-ray Sources".
* Instrument specifications for the *Athena* WFI and X-IFU have been added to the instrument registry.
* A test suite infrastructure has been added to SOXS, which runs automatically on GitHub when changes
  are made to the source code. 
* Simulating backgrounds without an input source is now possible by providing ``None`` to 
  :func:`~soxs.instrument.instrument_simulator` or ``"None"`` to the ``instrument_simulator`` command 
  line script (see :ref:`cmd-instrument`).
* The default astrophysical background in SOXS was not identical to the advertised input spectrum, which
  has been fixed.
* The options for dealing with background have been restricted. Backgrounds can now only be turned on 
  and off. The keyword arguments to :func:`~soxs.instrument.instrument_simulator` for dealing with 
  background have been correspondingly modified (see :ref:`instrument` and :ref:`cmd-instrument`). This
  is a backwards-incompatible change.
* The default version of APEC in :class:`~soxs.spectra.ApecGenerator` is now version 2.0.2, to match
  XSPEC. 
* A new option has been added to the instrument specification to turn dithering on and off by default
  for a given instrument. Please change instrument specification JSON files accordingly.
* Instead of the plate scale, the instrument field of view is specified in the instrument specification,
  and the plate scale is calculated from this and the number of pixels. Please change instrument 
  specification JSON files accordingly.

Version 0.3.1
-------------

This is a bugfix release.

* The RMF for the HDXI was updated so that the binning between it and the HDXI ARFs is consistent.
* Various small edits to the documentation were made.

Version 0.3.0
-------------

This version contains new features and bugfixes.

* An *Athena*-like microcalorimeter background is now the default particle background for all microcalorimeter models.
* All instrumental backgrounds now have a dependence on the focal length. The focal length is now an element of the
  instrument specification. 
* The names of the instruments in the instrument registry were made consistent with their associated keys.
* A convenience function, :meth:`~soxs.spectra.Spectrum.get_flux_in_band`, has been added. 
* A new method of generating a spectrum from an XSPEC script, :meth:`~soxs.spectra.Spectrum.from_xspec_script`, has been added.
* The :meth:`~soxs.spectra.Spectrum.from_xspec` method has been renamed to :meth:`~soxs.spectra.Spectrum.from_xspec_model`. 
* Removed unnecessary commas between coordinate values from the examples in :ref:`cmd-spatial`. 
* Added a new capability to create a SIMPUT file from an ASCII table of RA, Dec, and energy, 
  in the ``make_phlist_from_ascii`` command-line script.
* Added a new class for creating rectangle/line-shaped sources, :class:`~soxs.spatial.RectangleModel`, and a corresponding
  command-line script, ``make_rectangle_source``. 
* The signature of ``write_photon_list`` has changed to accept a ``flux`` argument instead of exposure time and area.

Version 0.2.1
-------------

This is a bugfix release.

* The supporting files (ARFs, RMFs, spectral files, etc.) were not being bundled properly in previous versions. 

Version 0.2.0
-------------

This version contains new features.

* New ARFs corresponding to various configurations of the mirrors have been added and the old ARFs have been
  removed (November 1st, 2016).
* Documentation now includes references to ways of getting help and the license.

Version 0.1.1
-------------

This is solely a bugfix release.

* Fixed a bug where the dither did not have the correct width.
* Fixed a bug for cases with no dithering.
* Various minor improvements to the documentation