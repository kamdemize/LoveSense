#pragma checksum "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\VerifiyMessage.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "f86ca455cc37499ad5c4098afc888e1cc82de857"
// <auto-generated/>
#pragma warning disable 1591
#pragma warning disable 0414
#pragma warning disable 0649
#pragma warning disable 0169

namespace LoveSense.Presentation.Web.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using LoveSense.Presentation.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using LoveSense.Presentation.Web.Shared;

#line default
#line hidden
#nullable disable
    [Microsoft.AspNetCore.Components.RouteAttribute("/VerifyMessage")]
    public partial class VerifiyMessage : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
        }
        #pragma warning restore 1998
#nullable restore
#line 33 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\VerifiyMessage.razor"
       
    private Message Message = new Message();
    private ResponseVerify result = new ResponseVerify();
    private string descriptionResponse;

    void FormSubmitted(EditContext editContext)
    {
        //ResponseVerify result = await MessageVerificator.VerifyAsync(Message.Text).Result;

        var task = Task.Run(async () => await MessageVerificator.VerifyAsync(Message.Text));
        var result = task.Result;

        if (result.Verdict)
            descriptionResponse = $"Your lover are probably a good one at {result.Score.ToString("P")}";
        else
            descriptionResponse = $"Your lover are probably a wron one at {result.Score.ToString("P")}";
    }

#line default
#line hidden
#nullable disable
        [global::Microsoft.AspNetCore.Components.InjectAttribute] private IMessageVerificator MessageVerificator { get; set; }
    }
}
#pragma warning restore 1591