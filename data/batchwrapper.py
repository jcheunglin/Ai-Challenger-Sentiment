#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/28 10:34
"""
import torch

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x &amp;amp;amp;amp;lt;g class="gr_ gr_3178 gr-alert gr_spell gr_inline_cards gr_disable_anim_appear ContextualSpelling ins-del" id="3178" data-gr-id="3178"&amp;amp;amp;amp;gt;and y&amp;amp;amp;amp;lt;/g&amp;amp;amp;amp;gt;

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:
                y = torch.cat([getattr(batch, feat).unsqueeze(1)+2 for feat in self.y_vars], dim=1)
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)

