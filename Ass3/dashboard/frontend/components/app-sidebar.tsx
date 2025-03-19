"use client"

import * as React from "react"
import { Inbox } from "lucide-react"

import { NavUser } from "@/components/nav-user"
import { Label } from "@/components/ui/label"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar"
import { Switch } from "@/components/ui/switch"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"


// This is sample data
const data = {
  user: {
    name: "shadcn",
    phoneNumber: "m@example.com",
    avatar: "/avatars/shadcn.jpg",
  },
  navMain: [
    {
      title: "Patient Info",
      url: "#",
      icon: Inbox,
      isActive: true,
    }
  ],
  mails: [
    {
      name: "Emma Roids",
      phoneNumber: "296-676-4892",
      lastCall: "03/09/2025",
      urgency: 9,
  },
  {
      name: "Phil McCracken",
      phoneNumber: "344-870-5101",
      lastCall: "03/09/2025",
      urgency: 3,
  },
  {
      name: "Jenna Tolls",
      phoneNumber: "216-867-4688",
      lastCall: "02/28/2025",
      urgency: 3,
  },
  {
      name: "Gloria Stits",
      phoneNumber: "216-667-2895",
      lastCall: "03/13/2025",
      urgency: 2,
  },
  {
      name: "Mike Rotch",
      phoneNumber: "595-455-3169",
      lastCall: "02/20/2025",
      urgency: 1,
  },
  {
      name: "Oliver Closehoff",
      phoneNumber: "398-572-1234",
      lastCall: "03/02/2025",
      urgency: 7,
  },
  {
      name: "Wayne Kerr",
      phoneNumber: "583-214-7654",
      lastCall: "02/25/2025",
      urgency: 6,
  },
  {
      name: "Ben Dover",
      phoneNumber: "784-563-8291",
      lastCall: "03/10/2025",
      urgency: 8,
  },
  {
      name: "Mike Hunt",
      phoneNumber: "609-478-9632",
      lastCall: "02/18/2025",
      urgency: 4,
  },
  {
      name: "Drew Peacock",
      phoneNumber: "432-671-3498",
      lastCall: "03/06/2025",
      urgency: 5,
  },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  // Note: I'm using state to show active item.
  // IRL you should use the url/router.
  const [activeItem, setActiveItem] = React.useState(data.navMain[0])
  const [mails, setMails] = React.useState(data.mails)
  const { setOpen } = useSidebar()

  return (
    <Sidebar
      collapsible="icon"
      className="overflow-hidden [&>[data-sidebar=sidebar]]:flex-row"
      {...props}
    >

      {/* This is the second sidebar */}
      {/* We disable collapsible and let it fill remaining space */}
      <Sidebar collapsible="none" className="hidden flex-1 md:flex">
        <SidebarHeader className="gap-3.5 border-b p-4">
          <div className="flex w-full items-center justify-between">
            <div className="text-base font-medium text-foreground">
              {activeItem?.title}
            </div>
            <Label className="flex items-center gap-2 text-sm">
              <DropdownMenu>
                <DropdownMenuTrigger>Filter By</DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem>Urgency</DropdownMenuItem>
                  <DropdownMenuItem>Last Call</DropdownMenuItem>
                  <DropdownMenuItem>Idk, tickle my fancy! 😳👉👈</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

            </Label>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup className="px-0">
            <SidebarGroupContent>
              {mails.map((mail) => (
                <a
                  href="#"
                  key={mail.phoneNumber}
                  className="flex flex-col items-start gap-2 whitespace-nowrap border-b p-4 text-sm leading-tight last:border-b-0 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                >
                  <div className="flex w-full items-center gap-2">
                    <span>{mail.name}</span>{" "}
                    <span className="ml-auto text-xs">Urgency: {mail.urgency}</span>
                  </div>
                  <span className="font-medium">Last Call: {mail.lastCall}</span>
                  
                  
                </a>
              ))}
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>
    </Sidebar>
  )
}
