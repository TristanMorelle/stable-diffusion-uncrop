import gradio as gr
import frontend.ui_functions as uifn


class Resize_Tab():

    def __init__(self):
        """
        defaults
        ================================================================================================================
        """
        self.mask_dither_opacity_nr = None
        self.border_thickness_nr = None
        self.auto_border_size_check = None
        self.auto_border_ratio_nr = None
        self.comp_img_edge_check = None
        self.fill_bg_check = None
        self.max_init_it_nr = None
        self.max_init_dist_nr = None
        self.max_opt_nr = None
        self.max_opt_dist_nr = None
        self.blur_bg_amount_nr = None
        self.comp_blur_nr = None
        self.debug_check = None
        self.save_resize_preset_name = None
        self.save_resize_preset_btn = None
        self.resize_img_button = None
        self.mask_bright_nr = None
        self.mask_bright_nr = None
        self.mask_blur_range_nr = None
        self.mask_final_check = None
        self.bg_bg_color_dropdown = None
        self.bg_picker = None
        self.sample_nr = None
        self.fill_border_size = None
        self.shrink_faded_edges_nr = None
        self.pre_fill_square = None
        self.resize_sequence_text = None
        self.use_resize_sequence_check = None
        self.resize_preset_dropdown = None
        self.pre_fill_picker = None

        # dicts
        self.param_ui_dict = dict()  # dict holding the gradui ui mapped to resize params
        self.picker_dict = dict()  # dict mapping color pickers to params
        self.online_settings_dict = dict()  # dict holding the current online settings (updated by offline)
        # dict holding the current online color picker settings (updated by offline)
        self.online_pickers_settings_dict = dict()

        # other needed ui elements for signals
        self.img2img_resize = None
        self.img2img_image_editor = None
        self.img2img_image_mask = None
        self.img2img_image_editor_mode = None
        self.img2img_width = None
        self.img2img_height = None
        self.tabs = None

    def build_resize_tab(self, img2img_resize_modes, img2img_defaults):
        """
        build resize ui tob in gradio
        ================================================================================================================
        """
        with gr.Tabs():
            with gr.TabItem("Resize options"):

                self.use_resize_sequence_check = gr.Checkbox(value=False,
                                                             interactive=True,
                                                             label='Resize using Sequence')

                self.resize_sequence_text = gr.Textbox(label="Resize Preset sequence (preset1;preset2;preset1)",
                                                       value="", visible=True)

                with gr.Row():
                    self.img2img_resize = gr.Dropdown(label="Resize mode",
                                                      choices=uifn.img_p.get_resize_funtions(),
                                                      type="index",
                                                      value=img2img_resize_modes[
                                                          img2img_defaults['resize_mode']], visible=True)

                    self.resize_preset_dropdown = gr.Dropdown(label="Resize Preset",
                                                              choices=self.get_presets(),
                                                              value="Default", visible=True)



                with gr.Accordion("Advanced resize options", open=False):
                    with gr.Row():
                        with gr.Column():
                            # repeat edges
                            self.pre_fill_square = gr.Dropdown(label='Pre-fill content bounding box',
                                                               value='auto_edge',
                                                               choices=['None',
                                                                        'auto_edge',
                                                                        'auto',
                                                                        'picker'], interactive=True)

                            self.pre_fill_picker = gr.ColorPicker(label='bg color', visible=False,
                                                                  show_label=True,
                                                                  interactive=True).style(
                                border=(0, 0, 0, 0),
                                container=True)

                            self.shrink_faded_edges_nr = gr.Number(value=0,
                                                                   precision=0,
                                                                   interactive=True,
                                                                   label='pre-Shrink faded edges')

                            self.fill_border_size = gr.Number(value=-1,
                                                              label='Fill border size (-1 is auto)',
                                                              precision=0,
                                                              interactive=True)

                            self.sample_nr = gr.Number(value=1,
                                                       label='border sample size',
                                                       precision=0,
                                                       interactive=True,
                                                       )

                            self.bg_bg_color_dropdown = gr.Dropdown(label='BG color',
                                                                    value='auto_edge',
                                                                    choices=['None',
                                                                             'auto_edge',
                                                                             'auto',
                                                                             'picker'],
                                                                    show_label=True,
                                                                    interactive=True)

                            self.bg_picker = gr.ColorPicker(label='bg color', visible=False,
                                                            show_label=True,
                                                            interactive=True).style(border=(0, 0, 0, 0),
                                                                                    container=True)

                        with gr.Column():
                            self.mask_final_check = gr.Checkbox(value=True,
                                                                interactive=True,
                                                                label='mask final image with Bg')

                            # update value
                            self.mask_blur_range_nr = gr.Number(value=100,
                                                                interactive=True,
                                                                precision=0,
                                                                label='Mask blur range (can get slow)')

                            self.mask_bright_nr = gr.Number(value=2,
                                                            interactive=True,
                                                            precision=2,
                                                            label='mask brightening multiplier')

                            self.mask_dither_opacity_nr = gr.Number(value=0.01,
                                                                    precision=3,
                                                                    interactive=True,
                                                                    label='mask dither opacity')

                        with gr.Column():
                            # scatter fill

                            self.border_thickness_nr = gr.Number(value=0,
                                                                 interactive=True,
                                                                 precision=0,
                                                                 label='border thickness, (0 = natual spead)')

                            self.auto_border_size_check = gr.Checkbox(value=False,
                                                                      interactive=True,
                                                                      label='use auto border size to content')

                            self.auto_border_ratio_nr = gr.Number(value=20,
                                                                  interactive=True,
                                                                  precision=0,
                                                                  label='1/x auto border ratio to content (1 = full)')

                            self.comp_img_edge_check = gr.Checkbox(value=True,
                                                                   interactive=True,
                                                                   label='comp img edge with small fade')

                            self.fill_bg_check = gr.Checkbox(value=True,
                                                             interactive=True,
                                                             label='fill bg')
                        with gr.Column():
                            # global scatter

                            self.max_init_it_nr = gr.Number(value=50,
                                                            interactive=True,
                                                            precision=0,
                                                            label='initial iterations to scatter')

                            self.max_init_dist_nr = gr.Number(value=2000,
                                                              interactive=True,
                                                              precision=0,
                                                              label='max initial distance from source px')

                            # local growth

                            self.max_opt_nr = gr.Number(value=50,
                                                        interactive=True,
                                                        precision=0,
                                                        label='optimize iterations')

                            self.max_opt_dist_nr = gr.Number(value=150,
                                                             interactive=True,
                                                             precision=0,
                                                             label='max optimize distance to grow')

                            self.blur_bg_amount_nr = gr.Number(value=0,
                                                               interactive=True,
                                                               precision=0,
                                                               label='blur bg result before comp')

                            self.comp_blur_nr = gr.Number(value=1,
                                                          interactive=True,
                                                          precision=0,
                                                          label='blur result edge on bg')

                            with gr.Row():
                                with gr.Column():
                                    self.debug_check = gr.Checkbox(value=False,
                                                                   interactive=True,
                                                                   label='Debug image creation')
                                    user = uifn.img_p.get_user_name()
                                    self.save_resize_preset_name = gr.Textbox('{}_preset'.format(user),
                                                                              label='preset name')
                                    self.save_resize_preset_btn = gr.Button("Save Resize Preset")

                self.resize_img_button = gr.Button("Resize")
                return self.img2img_resize

    def gather_vars(self,
                    img2img_image_editor,
                    img2img_image_mask,
                    img2img_image_editor_mode,
                    img2img_width,
                    img2img_height,
                    tabs):
        """
        set input and output signals on resize procedures
        ================================================================================================================
        """
        # transfer to class vars
        self.img2img_image_editor = img2img_image_editor
        self.img2img_image_mask = img2img_image_mask
        self.img2img_image_editor_mode = img2img_image_editor_mode
        self.img2img_width = img2img_width
        self.img2img_height = img2img_height
        self.tabs = tabs

    def connect_signals(self):
        """
        set input and output signals on resize procedures
        ================================================================================================================
        """
        # button pressed will uncrop the image and update the ui
        self.resize_img_button.click(self.crop_btn_procedure,
                                     [self.img2img_resize,
                                      self.img2img_image_editor,
                                      self.img2img_image_mask,
                                      self.img2img_image_editor_mode,
                                      self.img2img_width,
                                      self.img2img_height],

                                     [self.img2img_image_editor,
                                      self.img2img_image_mask,
                                      self.tabs]
                                     )

        # save button
        self.save_resize_preset_btn.click(self.save_resize_preset_procedure, [self.save_resize_preset_name],
                                          [self.resize_preset_dropdown])

        # load resize when dropdown toggled
        self.resize_preset_dropdown.change(self.load_resize_preset_procedure,
                                           [self.resize_preset_dropdown, self.img2img_resize],
                                           self.get_resize_ui_elements())

        # load resize when dropdown toggled
        self.use_resize_sequence_check.change(self.get_preset_choices,
                                           [self.resize_preset_dropdown],[self.resize_preset_dropdown])


        # change resize mode will change the ui settings
        self.img2img_resize.change(self.load_resize_mode_procedure, [self.img2img_resize],
                                   self.get_resize_ui_elements())

        # toggle color pickers when dropdown is 'picker'
        func_a = lambda x: self.check_picker(value=x, dropdown=self.pre_fill_picker)
        self.pre_fill_square.change(func_a, [self.pre_fill_square],
                                    [self.pre_fill_picker])

        func_b = lambda x: self.check_picker(value=x, dropdown=self.bg_picker)
        self.bg_bg_color_dropdown.change(func_b, [self.bg_bg_color_dropdown],
                                         [self.bg_picker])

    def init_ui(self, debug=False):
        """
        fix ui to accommodate the default settings and install a auto updater from online to offline
        ================================================================================================================
        """
        # fix init
        self.map_resize_params_to_ui()
        self.map_pickers()

        self.get_init_configs()
        if debug:
            for i in self.online_settings_dict:
                print(i.label)
        self.install_value_updater()
        self.init_visibility()

    def check_picker(self, value, dropdown=None):
        """
        hide pickers when 'picker' is not selected string
        ================================================================================================================
        """
        if not dropdown:
            raise RuntimeError('no dropdown ui given, ', dropdown)

        if dropdown not in self.online_pickers_settings_dict:
            raise RuntimeError('dropdown not found in dict', dropdown, self.online_pickers_settings_dict.keys())

        if value == 'picker':
            self.online_pickers_settings_dict[dropdown]['visible'] = True
            return gr.update(visible=True)
        else:
            self.online_pickers_settings_dict[dropdown]['visible'] = False
            return gr.update(visible=False)

    def get_map_resize_params_to_ui(self):
        """
        map gr ui to resize func params the first call, or return existing if already created
        ================================================================================================================
        """
        return self.param_ui_dict if self.param_ui_dict else self.map_resize_params_to_ui()

    def map_resize_params_to_ui(self):
        """
        map gr ui to resize func params the first call
        ================================================================================================================
        """
        # create dict with ui mapping to params

        # presets
        self.param_ui_dict['resize_sequence'] = self.resize_sequence_text
        self.param_ui_dict['use_resize_sequence'] = self.use_resize_sequence_check

        # repeat edges
        self.param_ui_dict['resize_mode'] = self.img2img_resize
        self.param_ui_dict['pre_fill_square'] = self.pre_fill_square
        self.param_ui_dict['width'] = self.img2img_width
        self.param_ui_dict['height'] = self.img2img_height
        self.param_ui_dict['shrink_faded_edges'] = self.shrink_faded_edges_nr
        self.param_ui_dict['fill_border_size'] = self.fill_border_size
        self.param_ui_dict['sample'] = self.sample_nr
        self.param_ui_dict['bg_color'] = self.bg_bg_color_dropdown
        self.param_ui_dict['mask_final'] = self.mask_final_check
        self.param_ui_dict['mask_blur_range'] = self.mask_blur_range_nr
        self.param_ui_dict['mask_bright'] = self.mask_bright_nr
        self.param_ui_dict['mask_dither_opacity'] = self.mask_dither_opacity_nr

        # scatter fill
        self.param_ui_dict['border_thickness'] = self.border_thickness_nr
        self.param_ui_dict['auto_border_size'] = self.auto_border_size_check
        self.param_ui_dict['auto_border_ratio'] = self.auto_border_ratio_nr
        self.param_ui_dict['comp_img_edge'] = self.comp_img_edge_check
        self.param_ui_dict['fill_bg'] = self.fill_bg_check
        # scatter
        self.param_ui_dict['max_init_it'] = self.max_init_it_nr
        self.param_ui_dict['max_init_dist'] = self.max_init_dist_nr
        # local growth
        self.param_ui_dict['max_opt'] = self.max_opt_nr
        self.param_ui_dict['max_opt_dist'] = self.max_opt_dist_nr
        self.param_ui_dict['blur_bg_amount'] = self.blur_bg_amount_nr
        self.param_ui_dict['mask_blur_range'] = self.mask_blur_range_nr
        self.param_ui_dict['comp_blur'] = self.comp_blur_nr
        # debug
        self.param_ui_dict['debug'] = self.debug_check

        return self.param_ui_dict

    def get_default_values(self, resize_mode=None):
        """
        get default values and set to ui
        ================================================================================================================
        """
        if not resize_mode:
            resize_mode = self.online_settings_dict[self.img2img_resize]['value']

        defaults = uifn.img_p.get_default_presets(preset_name=resize_mode)
        defaults['resize_mode'] = resize_mode
        return defaults

    def get_map_pickers(self):
        """
        map gr pickers ui to resize func params the first call, or return existing if already created
        ================================================================================================================
        """
        return self.picker_dict if self.picker_dict else self.map_pickers()

    def map_pickers(self):
        """
        map additional color pickers
        ================================================================================================================
        """
        if not self.picker_dict:
            self.picker_dict['bg_color'] = self.bg_picker
            self.picker_dict['pre_fill_square'] = self.pre_fill_picker
        return self.picker_dict

    def update_value(self, func):
        """
        abstract function to set offline ui value from input stream, so offline is up to date with live value
        ================================================================================================================
        """
        func.change(lambda x: self.set_value(func, x),
                    inputs=[func], outputs=[])

    def update_picker_value(self, func):
        """
        abstract function to set offline ui value from input stream, so offline is up to date with live value
        ================================================================================================================
        """
        func.change(lambda x: self.set_picker_value(func, x),
                    inputs=[func], outputs=[])

    def set_value(self, func, x):
        """
        set value on ui and local settings dict
        ================================================================================================================
        """
        # convert index to string
        if func == self.img2img_resize and type(x) in (int, float):
            x = self.img2img_resize.choices[x]

        self.online_settings_dict[func]['value'] = x

        # update offline ui with online value
        return setattr(func, 'value', x)

    def set_picker_value(self, func, x):
        """
        set picker value on ui and local settings dict
        ================================================================================================================
        """
        self.online_pickers_settings_dict[func]['value'] = x
        # upcate offline ui with online value
        return setattr(func, 'value', x)

    def get_init_configs(self):
        """
        get all init configs from ui
        ================================================================================================================
        """
        params_ui = self.get_map_resize_params_to_ui()
        pickers = self.get_map_pickers()
        for p, ui in params_ui.items():
            self.online_settings_dict[ui] = ui.get_config()

        for p, ui in pickers.items():
            self.online_pickers_settings_dict[ui] = ui.get_config()

    def install_value_updater(self):
        """
        function to install 'self updater' on ui elements
        update: initially thought to use the offline gr functions by keeping the value up to date using "change",
        but now switched to the 'online dict' for saving purposes ...
        TODO: remove self value updater and only use online dict? doesn't do any harm atm
        ================================================================================================================
        """
        params_ui = self.get_map_resize_params_to_ui()
        pickers = self.get_map_pickers()

        for p, ui in params_ui.items():
            self.update_value(func=ui)

        for p, ui in pickers.items():
            self.update_picker_value(func=ui)

    def get_resize_settings(self, debug=False):
        """
        get resize ui settings from ui and map to settings dict
        ================================================================================================================
        """
        settings_dict = dict()
        params_ui = self.get_map_resize_params_to_ui()
        pickers = self.get_map_pickers()
        for p, ui in params_ui.items():
            # don't get ui element if not visible
            if not self.online_settings_dict[ui]['visible']:
                if debug:
                    print('skipping', p)
                continue
            value = self.online_settings_dict[ui]['value']

            if p == "resize_mode" and type(value) in (int, float):
                value = self.img2img_resize.choices[value]

            settings_dict[p] = value

            if p in pickers and value == 'picker':
                # pass color picker value
                settings_dict[p] = pickers[p].value

        if debug:
            print(50 * '=')
            print(settings_dict)
            print(50 * '=')

        return settings_dict

    def set_resize_settings(self, settings_dict, debug=False):
        """
        get resize settings from dict and set to offline ui
        ================================================================================================================
        """
        params_ui = self.get_map_resize_params_to_ui()
        pickers = self.get_map_pickers()
        for p, ui in params_ui.items():
            if p not in settings_dict:
                # not setting the ui as it is not part of this preset
                if debug:
                    print(p, 'not found as possible ui to set')
                continue

            # should not be affected by setter
            elif p in self.exceptions_to_set():
                if debug:
                    print(p, 'not setting')
                continue

            # update pickers and set the value
            value = settings_dict[p]
            if p in pickers and type(value) in (list, tuple):
                # first set the picker color
                self.online_pickers_settings_dict[ui]['value'] = value
                # the set drop down to picker
                value = 'picker'

            # if int is passed as resize mode, update to string
            elif p == "resize_mode" and type(value) in (int, float):
                value = self.img2img_resize.choices[value]

            # if tuple with all the same values as for all sides of images reform to single int to pass to ui
            elif type(value) in (list, tuple) and all([value[0] == i for i in value]):
                value = value[0]

            # update online ui
            self.online_settings_dict[ui]['value'] = value

            if debug:
                print('seting resize ==>', p, value)

        return settings_dict

    def exceptions_to_set(self):
        """
        should not be affected by ui setters
        ================================================================================================================
        """
        return ['width', 'height']

    def show_hide_ui(self, settings_dict):
        """
        get a dict with visibility settings based on the provided settings dict
        ================================================================================================================
        """
        # var/ui mapping
        ui_dict = self.get_map_resize_params_to_ui()

        # if it's not in the settings dict, hide it, else show
        return_dict = dict()
        for p, ui in ui_dict.items():
            if p in settings_dict:
                return_dict[ui_dict[p]] = True

            elif p in self.exceptions_to_set():  # always show:
                return_dict[ui_dict[p]] = True

            else:
                return_dict[ui_dict[p]] = False

        # update dict with online settings
        for ui, v in return_dict.items():
            self.online_settings_dict[ui]['visible'] = v

        return return_dict

    def show_hide_picker_ui(self, settings_dict):
        """
        get a dict with visibility settings based on the provided settings dict for the pickers
        ================================================================================================================
        """
        # reverse mapping
        ui_picker_dict = self.get_map_pickers()

        # show/hide pickers
        return_dict = dict()
        for p, ui in ui_picker_dict.items():
            if p in settings_dict and settings_dict[p] == 'picker':
                return_dict[ui_picker_dict[p]] = True
            else:
                return_dict[ui_picker_dict[p]] = False

        # update online picker dict with visibility
        for ui, v in return_dict.items():
            if ui in self.online_pickers_settings_dict:
                self.online_pickers_settings_dict[ui]['visible'] = v

    def init_visibility(self):
        """
        get/set init visibility of ui
        ================================================================================================================
        """
        self.show_hide_ui(settings_dict=self.get_default_values())
        self.show_hide_picker_ui(settings_dict=self.get_default_values())
        # ui is not yet created so set it on ui for init
        for ui, v_dict in self.online_settings_dict.items():
            ui.visible = v_dict['visible']

        for ui, v_dict in self.online_pickers_settings_dict.items():
            ui.visible = v_dict['visible']

    def save_resize_preset_procedure(self, preset_name):
        """
        save resize settings dict as preset to disk
        ================================================================================================================
        """
        if preset_name == 'Default':
            return gr.update(value=preset_name)

        preset_name = preset_name.replace(' ', '_')
        get_settings_dict = self.get_resize_settings()
        uifn.img_p.save_preset(preset=preset_name, data=get_settings_dict)
        choices = self.get_presets()
        return gr.update(choices=choices, value=preset_name)

    def load_resize_preset_procedure(self, preset_name, resize_mode):
        """
        load resize settings dict and set to ui
        ================================================================================================================
        """
        # load preset pickle or get defaults
        if preset_name == 'Default':
            data = self.get_default_values(resize_mode=resize_mode)
        else:
            data = uifn.img_p.load_preset(preset=preset_name)

        if data:
            # first set online ui elements with value
            self.set_resize_settings(settings_dict=data)
        # then update from values
        return self.get_resize_ui_update_list(settings_dict=data)

    def get_presets(self):
        """
        get presets from disk + default
        ================================================================================================================
        """
        return ['Default'] + uifn.img_p.list_presets()

    def get_preset_choices(self, value):
        """
        load presets to ui choices
        ================================================================================================================
        """
        return gr.update(choices=self.get_presets(), value=value)

    def load_resize_mode_procedure(self, resize_mode):
        """
        load defaults for when resize mode is chosen
        ================================================================================================================
        """
        data = self.get_default_values(resize_mode=resize_mode)
        if data:
            # first set ui elements with value
            self.set_resize_settings(settings_dict=data)
        # then update from values
        return self.get_resize_ui_update_list(settings_dict=data)

    def get_resize_ui(self):
        """
        get list of ui elements without pickers
        ================================================================================================================
        """
        return list(self.get_map_resize_params_to_ui().values())

    def get_pickers_ui(self):
        """
        get list of only ui pickers associated with other ui settings
        ================================================================================================================
        """
        return list(self.get_map_resize_params_to_ui().values())

    def get_preset_ui(self):
        """
        get preset ui
        ================================================================================================================
        """
        return [self.resize_preset_dropdown]

    def get_resize_ui_elements(self):
        """
        get list of all ui elements
        ================================================================================================================
        """
        params_ui = self.get_resize_ui()
        pickers = self.get_pickers_ui()
        # preset = self.get_preset_ui()
        return params_ui + pickers

    def get_resize_ui_update_list(self, settings_dict):
        """
        get value and vis update list to set/update ui
        ================================================================================================================
        """
        # dict with key: ui_vis, value: True/False
        # get visible ui's
        vis = self.show_hide_ui(settings_dict=settings_dict)
        update_list = []
        for i in self.get_resize_ui_elements():
            if i in vis:
                # update value and vis
                v = self.online_settings_dict[i]['value']
                update_list.append(gr.update(value=v, visible=vis[i]))
            else:
                # just repeat the value to keep the consistent order of gr updates
                v = self.online_settings_dict[i]['value']
                update_list.append(gr.update(value=v))
        return update_list

    def crop_btn_procedure(self, *args, **kwargs):
        """
        fetch settings dict and pass as kwargs to resize function
        ================================================================================================================
        """
        kwargs['resize_settings'] = self.get_resize_settings()
        return uifn.crop_btn_procedure(*args, **kwargs)
